import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
import numpy as np
from tqdm.auto import tqdm
from einops import reduce

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x): return x is not None
def default(val, d): return val if exists(val) else (d() if callable(d) else d)
def identity(t, *a, **k): return t

def extract(arr, timesteps, broadcast_shape):
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def normalize_to_neg_one_to_one(img): return img * 2 - 1
def unnormalize_to_zero_to_one(t): return (t + 1) * 0.5
def mean_flat(tensor): return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = next(o for o in (mean1, logvar1, mean2, logvar2) if isinstance(o, torch.Tensor))
    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor) for x in (logvar1, logvar2)]
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2)**2) * torch.exp(-logvar2))

def discretized_gaussian_log_likelihood(z, mean, log_std):
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps, max_beta=0.01):
    return torch.linspace(1e-4, max_beta, timesteps)

class ImageProjector(nn.Module):
    def __init__(self, input_dim=1536, img_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, img_dim)
        )
    def forward(self, x):
        return self.proj(x)

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, q, kv):
        Q = self.to_q(q)
        K = self.to_k(kv)
        V = self.to_v(kv)
        attn = torch.softmax(torch.bmm(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.bmm(attn, V)
        return self.norm(q + self.out_proj(out))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class ST_Diffusion_MM(nn.Module):
    def __init__(self, n_steps, dim, num_units=64, self_condition=False,
                 condition=True, cond_dim=64, img_dim=64):
        super().__init__()
        self.channels = 1
        self.self_condition = self_condition
        self.condition = condition
        sinu_pos_emb = SinusoidalPosEmb(num_units)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb, nn.Linear(num_units, num_units), nn.GELU(), nn.Linear(num_units, num_units))
        self.linears_spatial = nn.ModuleList([
            nn.Linear(dim-1, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units)])
        self.linears_temporal = nn.ModuleList([
            nn.Linear(1, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units)])
        self.output_spatial  = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, dim-1))
        self.output_temporal = nn.Sequential(nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, 1))
        self.linear_t = nn.Sequential(nn.Linear(num_units*2, num_units), nn.ReLU(), nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, 2))
        self.linear_s = nn.Sequential(nn.Linear(num_units*2, num_units), nn.ReLU(), nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, 2))
        # enc_out is 128-d (temporal 64 + spatial 64), no joint component
        self.cond_all      = nn.Sequential(nn.Linear(cond_dim*2, num_units), nn.ReLU(), nn.Linear(num_units, num_units))
        self.cond_temporal = nn.ModuleList([nn.Linear(cond_dim, num_units) for _ in range(3)])
        self.cond_spatial  = nn.ModuleList([nn.Linear(cond_dim, num_units) for _ in range(3)])
        # img_cond (mu_fused) injected additively, not via cross-attention
        self.img_proj = nn.Linear(img_dim, num_units)

    def _split_cond(self, cond):
        h = cond.shape[-1] // 2
        return cond[:,:,:h], cond[:,:,h:]

    def forward(self, x, t, x_self_cond=None, cond=None, img_cond=None):
        x_spatial  = x[:,:,1:].clone()
        x_temporal = x[:,:,:1].clone()
        cond_temporal, cond_spatial = self._split_cond(cond)
        cond_merged = self.cond_all(cond)
        t_emb = self.time_mlp(t).unsqueeze(1)
        cond_all = torch.cat((cond_merged, t_emb), dim=-1)
        alpha_s = F.softmax(self.linear_s(cond_all), dim=-1).squeeze(1).unsqueeze(2)
        alpha_t = F.softmax(self.linear_t(cond_all), dim=-1).squeeze(1).unsqueeze(2)
        img_kv = self.img_proj(img_cond) if img_cond is not None else torch.zeros_like(t_emb)
        for idx in range(3):
            x_spatial  = self.linears_spatial[2*idx](x_spatial)
            x_temporal = self.linears_temporal[2*idx](x_temporal)
            x_spatial  += t_emb
            x_temporal += t_emb
            x_spatial  += self.cond_spatial[idx](cond_spatial)
            x_temporal += self.cond_temporal[idx](cond_temporal)
            # additive satellite condition (text description: flat conditioning signal)
            x_spatial  += img_kv
            x_temporal += img_kv
            x_spatial  = self.linears_spatial[2*idx+1](x_spatial)
            x_temporal = self.linears_temporal[2*idx+1](x_temporal)
        x_spatial  = self.linears_spatial[-1](x_spatial)
        x_temporal = self.linears_temporal[-1](x_temporal)
        x_output = torch.cat((x_temporal, x_spatial), dim=1)
        x_output_t = (x_output * alpha_t).sum(dim=1, keepdim=True)
        x_output_s = (x_output * alpha_s).sum(dim=1, keepdim=True)
        return torch.cat((self.output_temporal(x_output_t), self.output_spatial(x_output_s)), dim=-1)

class GaussianDiffusion_MM(nn.Module):
    def __init__(self, model, *, seq_length, timesteps=1000, sampling_timesteps=None,
                 loss_type='l2', objective='pred_noise', beta_schedule='cosine',
                 p2_loss_weight_gamma=0., p2_loss_weight_k=1, ddim_sampling_eta=1.):
        super().__init__()
        self.model = model
        self.channels = model.channels
        self.self_condition = model.self_condition
        self.seq_length = seq_length
        self.objective = objective
        self.loss_type = loss_type
        betas = cosine_beta_schedule(timesteps) if beta_schedule == 'cosine' else linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.num_timesteps = int(betas.shape[0])
        self.sampling_timesteps = default(sampling_timesteps, self.num_timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < self.num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        rb = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        rb('betas', betas)
        rb('alphas_cumprod', alphas_cumprod)
        rb('alphas_cumprod_prev', alphas_cumprod_prev)
        rb('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        rb('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        rb('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        rb('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        rb('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        rb('posterior_variance', posterior_variance)
        rb('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=posterior_variance[1])))
        rb('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        rb('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        rb('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    @property
    def loss_fn(self):
        if self.loss_type == 'l2': return F.mse_loss
        if self.loss_type == 'l1': return F.l1_loss
        raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def q_posterior(self, x_start, x_t, t):
        pm  = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
               extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        pv  = extract(self.posterior_variance, t, x_t.shape)
        plv = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return pm, pv, plv

    def q_mean_variance(self, x_start, t):
        mean     = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_var  = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_var

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, cond=None, img_cond=None):
        model_output = self.model(x, t, x_self_cond, cond=cond, img_cond=img_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = maybe_clip(self.predict_start_from_noise(x, t, pred_noise))
        elif self.objective == 'pred_x0':
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True, cond=None, img_cond=None):
        preds = self.model_predictions(x, t, x_self_cond, cond=cond, img_cond=img_cond)
        x_start = preds.pred_x_start
        if clip_denoised: x_start.clamp_(-1., 1.)
        model_mean, pv, plv = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, pv, plv, x_start

    @torch.no_grad()
    def p_sample(self, x, t, x_self_cond=None, cond=None, img_cond=None):
        b = x.shape[0]
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, cond=cond, img_cond=img_cond)
        noise = torch.randn_like(x) if t > 0 else 0.
        return model_mean + (0.5 * model_log_variance).exp() * noise, x_start

    @torch.no_grad()
    def ddim_sample(self, shape, cond=None, img_cond=None, clip_denoised=True):
        batch, device = shape[0], self.betas.device
        times = torch.linspace(-1, self.num_timesteps-1, steps=self.sampling_timesteps+1)
        time_pairs = list(zip(reversed(times.int().tolist()[:-1]), reversed(times.int().tolist()[1:])))
        img = torch.randn(shape, device=device)
        x_start = None
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            preds = self.model_predictions(img, time_cond, self_cond, clip_x_start=clip_denoised, cond=cond, img_cond=img_cond)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
            if time_next < 0:
                img = x_start; continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = self.ddim_sampling_eta * ((1 - alpha/alpha_next) * (1-alpha_next) / (1-alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * torch.randn_like(img)
        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def p_sample_loop(self, shape, cond=None, img_cond=None):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, cond=cond, img_cond=img_cond)
        return unnormalize_to_zero_to_one(img)

    @torch.no_grad()
    def sample(self, batch_size=16, cond=None, img_cond=None):
        fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return fn((batch_size, self.channels, self.seq_length), cond=cond, img_cond=img_cond)

    def p_losses(self, x_start, t, noise=None, cond=None, img_cond=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x, t, None, cond=cond, img_cond=img_cond)
        target = noise if self.objective == 'pred_noise' else x_start
        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def _vb_terms_bpd(self, x_start, x_t, t, clip_denoised=True, cond=None, img_cond=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x=x_t, t=t, clip_denoised=clip_denoised, cond=cond, img_cond=img_cond)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl_all = mean_flat(kl) / np.log(np.e)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, model_mean, 0.5*model_log_variance)
        decoder_nll_all = mean_flat(decoder_nll) / np.log(np.e)
        kl_temporal = mean_flat(kl[:,:,:1]) / np.log(np.e)
        kl_spatial  = mean_flat(kl[:,:,-(self.seq_length-1):]) / np.log(np.e)
        decoder_nll_temporal = mean_flat(decoder_nll[:,:,:1]) / np.log(np.e)
        decoder_nll_spatial  = mean_flat(decoder_nll[:,:,-(self.seq_length-1):]) / np.log(np.e)
        output          = torch.where(t==0, decoder_nll_all,      kl_all)
        output_temporal = torch.where(t==0, decoder_nll_temporal, kl_temporal)
        output_spatial  = torch.where(t==0, decoder_nll_spatial,  kl_spatial)
        return output, output_temporal, output_spatial, pred_xstart

    def _prior_bpd(self, x_start):
        b = x_start.shape[0]
        t = torch.tensor([self.num_timesteps-1]*b, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.)
        return (mean_flat(kl_prior) / np.log(np.e),
                mean_flat(kl_prior[:,:,:1]) / np.log(np.e),
                mean_flat(kl_prior[:,:,-(self.seq_length-1):]) / np.log(np.e))

    def NLL_cal(self, x_start, cond, img_cond=None, noise=None, clip_denoised=True):
        x_start = normalize_to_neg_one_to_one(x_start)
        device, b = x_start.device, x_start.shape[0]
        vb_all, vb_t_all, vb_s_all = [], [], []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = torch.tensor([t]*b, device=device)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with torch.no_grad():
                vb, vbt, vbs, _ = self._vb_terms_bpd(x_start, x_t, t_batch, clip_denoised, cond=cond, img_cond=img_cond)
            vb_all.append(vb.unsqueeze(1))
            vb_t_all.append(vbt.unsqueeze(1))
            vb_s_all.append(vbs.unsqueeze(1))
        vb_all  = torch.sum(torch.cat(vb_all,  dim=-1), dim=-1)
        vb_t_all = torch.sum(torch.cat(vb_t_all, dim=-1), dim=-1)
        vb_s_all = torch.sum(torch.cat(vb_s_all, dim=-1), dim=-1)
        prior_all, prior_t, prior_s = self._prior_bpd(x_start)
        return (vb_all+prior_all).sum().item(), (vb_t_all+prior_t).sum().item(), (vb_s_all+prior_s).sum().item()

    def forward(self, img, cond, img_cond=None, *args, **kwargs):
        b, c, n, device = *img.shape, img.device
        assert n == self.seq_length
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, cond=cond, img_cond=img_cond, *args, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# Gaussian uncertainty heads
# ─────────────────────────────────────────────────────────────────────────────

class GaussianHead1D(nn.Module):
    """Linear head that produces (mu, sigma) from a flat embedding [N, D]."""
    def __init__(self, in_dim, out_dim, sigma_eps=1e-4):
        super().__init__()
        self.mu_head    = nn.Linear(in_dim, out_dim)
        self.sigma_head = nn.Linear(in_dim, out_dim)
        self.sigma_eps  = sigma_eps

    def forward(self, x):
        """x: [N, D]  →  (mu, sigma) each [N, D]"""
        return self.mu_head(x), F.softplus(self.sigma_head(x)) + self.sigma_eps


# ─────────────────────────────────────────────────────────────────────────────
# Mutual k-NN Alignment (3 modalities, event-level)
#
# Transformer_ST returns cat(enc_temporal, enc_loc, enc_joint) → 192-dim.
# We split this into three 64-dim modalities and align all three pairs:
#   T ↔ S,  T ↔ I,  S ↔ I
# Pairs touching the image modality are down-weighted by exp(-mean(sig_i)).
# ─────────────────────────────────────────────────────────────────────────────

class MutualKNNAlignment(nn.Module):
    def __init__(self, k=5, temp=0.1):
        super().__init__()
        self.k    = k
        self.temp = temp

    def _soft_knn(self, emb):
        """emb: [N, D]  →  soft adjacency [N, N]"""
        norm = F.normalize(emb, dim=-1)
        sim  = torch.mm(norm, norm.T) / self.temp
        val, idx = sim.topk(self.k, dim=-1)
        mask = torch.full_like(sim, float('-inf'))
        mask.scatter_(-1, idx, val)
        return F.softmax(mask, dim=-1)

    @staticmethod
    def _sym_kl(A, B):
        """Symmetric KL per row, averaged over N. Returns [N]."""
        eps = 1e-9
        kl_ab = (A * (A.clamp(eps).log() - B.clamp(eps).log())).sum(-1)
        kl_ba = (B * (B.clamp(eps).log() - A.clamp(eps).log())).sum(-1)
        return (kl_ab + kl_ba) / 2

    def forward(self, mu_t, mu_s, mu_i, sig_i):
        """
        mu_t:  [N, 64]  temporal embedding  (enc_nm[:, 0, :64])
        mu_s:  [N, 64]  spatial embedding   (enc_nm[:, 0, 64:128])
        mu_i:  [N, 64]  image embedding     (img_proj[:, 0, :])
        sig_i: [N, 64]  image uncertainty   (from image_gauss)
        Returns: scalar alignment loss
        """
        A_t = self._soft_knn(mu_t)
        A_s = self._soft_knn(mu_s)
        A_i = self._soft_knn(mu_i)

        # Image uncertainty weight: noisy tile → lower contribution
        w_i = torch.exp(-sig_i.mean(dim=-1))   # [N] ∈ (0, 1]

        kl_ts = self._sym_kl(A_t, A_s)          # T ↔ S  (both reliable)
        kl_ti = self._sym_kl(A_t, A_i) * w_i    # T ↔ I  (image down-weighted)
        kl_si = self._sym_kl(A_s, A_i) * w_i    # S ↔ I  (image down-weighted)

        return (kl_ts + kl_ti + kl_si).mean() / 3


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty-Guided MoE (3 modalities, event-level)
# ─────────────────────────────────────────────────────────────────────────────

class _ExpertMLP1D(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, sigma_eps=1e-4):
        super().__init__()
        self.fc        = nn.Linear(in_dim, hidden_dim)
        self.mu_head   = nn.Linear(hidden_dim, out_dim)
        self.sig_head  = nn.Linear(hidden_dim, out_dim)
        self.sigma_eps = sigma_eps

    def forward(self, z):
        h = F.relu(self.fc(z))
        return self.mu_head(h), F.softplus(self.sig_head(h)) + self.sigma_eps


class UGMoE(nn.Module):
    """
    Uncertainty-Guided MoE for three event-level modality embeddings:
      temporal (enc_nm[:64]), spatial (enc_nm[64:128]), image (img_proj).

    Each modality has its own router; gate weights are scaled by
    exp(-mean(sigma_m)) so unreliable modalities are automatically
    down-weighted per event.  Top-k routing + load-balance loss.
    """
    def __init__(self, in_dim=64, out_dim=64, n_experts=4, top_k=2,
                 expert_hidden=128, sigma_weight_form="exp", sigma_eps=1e-4,
                 use_uncertainty=True):
        super().__init__()
        self.n_experts = n_experts
        self.top_k     = top_k
        self.form      = sigma_weight_form
        self.use_uncertainty = use_uncertainty

        self.experts = nn.ModuleList([
            _ExpertMLP1D(in_dim, expert_hidden, out_dim, sigma_eps)
            for _ in range(n_experts)
        ])
        # One router per modality
        self.routers    = nn.ModuleList([nn.Linear(in_dim, n_experts) for _ in range(3)])
        self.mod_logits = nn.Parameter(torch.zeros(3))

    @staticmethod
    def _topk_gate(logits, k):
        vals, idx = logits.topk(k, dim=-1)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, idx, vals)
        return F.softmax(mask, dim=-1)

    def _sigma_scale(self, sigma):
        s = sigma.mean(dim=-1, keepdim=True)
        return torch.exp(-s) if self.form == "exp" else 1.0 / (1.0 + s)

    def forward(self, z_t, z_s, z_i, sig_t, sig_s, sig_i):
        """
        Each modality provides its own content embedding and uncertainty.
        z_t/z_s/z_i:   [N, D]  reparameterized embeddings (temporal/spatial/image)
        sig_t/sig_s/sig_i: [N, D]  per-modality uncertainties
        Returns:
            mu_out:    [N, D']
            sigma_out: [N, D']
            load_balance_loss: scalar
        """
        bal_terms = []
        mod_mus   = []
        mod_sigs  = []

        for z_m, router, sigma_m in zip(
                [z_t, z_s, z_i], self.routers, [sig_t, sig_s, sig_i]):

            # Each modality runs its own input through the shared expert pool
            exp_mus, exp_sigs = [], []
            for exp in self.experts:
                mu_e, sig_e = exp(z_m)
                exp_mus.append(mu_e)
                exp_sigs.append(sig_e)
            exp_mus  = torch.stack(exp_mus,  dim=1)  # [N, E, D']
            exp_sigs = torch.stack(exp_sigs, dim=1)

            logits = router(z_m)
            gate   = self._topk_gate(logits, self.top_k)
            if self.use_uncertainty:
                scale = self._sigma_scale(sigma_m)
                gate  = gate * scale
            gate   = gate / (gate.sum(-1, keepdim=True) + 1e-8)

            g     = gate.unsqueeze(-1)
            mu_m  = (g * exp_mus).sum(1)
            var_m = (g.pow(2) * exp_sigs.pow(2)).sum(1)
            mod_mus.append(mu_m)
            mod_sigs.append((var_m + 1e-6).sqrt())

            with torch.no_grad():
                f_tok = gate.gt(0).float().mean(0)
            f_prob = logits.softmax(-1).mean(0)
            bal_terms.append((f_tok * f_prob).sum())

        load_bal = torch.stack(bal_terms).mean()

        mod_sigma_mean = torch.stack([sig_t.mean(), sig_s.mean(), sig_i.mean()])
        if self.use_uncertainty:
            alpha = F.softmax(self.mod_logits - mod_sigma_mean, dim=0)  # [3]
        else:
            alpha = F.softmax(self.mod_logits, dim=0)  # [3], no sigma

        mu_out    = sum(alpha[m] * mod_mus[m]  for m in range(3))
        var_out   = sum(alpha[m].pow(2) * mod_sigs[m].pow(2) for m in range(3))
        sigma_out = (var_out + 1e-6).sqrt()

        return mu_out, sigma_out, load_bal


# ─────────────────────────────────────────────────────────────────────────────
# Full model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class Model_all_MM(nn.Module):
    def __init__(self, transformer, diffusion, img_projector, proj_dim=128,
                 n_experts=4, top_k=2, knn_k=5, knn_temp=0.1,
                 expert_hidden=128, sigma_eps=1e-4, use_uncertainty=True):
        super().__init__()
        self.transformer   = transformer
        self.diffusion     = diffusion
        self.img_projector = img_projector
        # MCL projection heads (kept from original)
        self.hist_proj = nn.Sequential(
            nn.Linear(64, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.img_proj_mcl = nn.Sequential(
            nn.Linear(64, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        # ── NEW: Gaussian uncertainty heads ───────────────────────────────────
        self.temporal_gauss = GaussianHead1D(64, 64, sigma_eps)
        self.spatial_gauss  = GaussianHead1D(64, 64, sigma_eps)
        self.image_gauss    = GaussianHead1D(64, 64, sigma_eps)
        # ── NEW: Mutual k-NN alignment ────────────────────────────────────────
        self.mutual_knn = MutualKNNAlignment(k=knn_k, temp=knn_temp)
        # ── NEW: Uncertainty-guided MoE ───────────────────────────────────────
        self.ugmoe = UGMoE(
            in_dim=64, out_dim=64,
            n_experts=n_experts, top_k=top_k,
            expert_hidden=expert_hidden, sigma_eps=sigma_eps,
            use_uncertainty=use_uncertainty,
        )
