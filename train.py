import os, sys, math, pickle, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import setproctitle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DSTPP import Transformer_ST as _Transformer_ST_Base

class Transformer_ST(_Transformer_ST_Base):
    """Drop the joint enc_output; return only temporal (64) + spatial (64) = 128-d."""
    def forward(self, event_loc, event_time):
        enc_output_all, non_pad_mask = super().forward(event_loc, event_time)
        return enc_output_all[:, :, :128], non_pad_mask
from DSTPP.Dataset import get_dataloader
from model_mm import ST_Diffusion_MM, GaussianDiffusion_MM, Model_all_MM, ImageProjector

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',          type=int, default=1234)
    p.add_argument('--dataset',       type=str, default='Citibike')
    p.add_argument('--total_epochs',  type=int, default=2000)
    p.add_argument('--alpha',         type=float, default=0.05,  help='MCL loss weight')
    p.add_argument('--alpha_knn',    type=float, default=0.1,   help='MutualKNN loss weight')
    p.add_argument('--alpha_bal',    type=float, default=0.01,  help='MoE load-balance loss weight')
    p.add_argument('--n_experts',    type=int,   default=4,     help='UGMoE number of experts')
    p.add_argument('--top_k',        type=int,   default=2,     help='UGMoE active experts per event')
    p.add_argument('--wo_uncertainty', action='store_true', help='ablation: disable sigma weighting in UGMoE')
    p.add_argument('--knn_k',        type=int,   default=5,     help='k for MutualKNN graph')
    p.add_argument('--batch_size',   type=int, default=64)
    p.add_argument('--timesteps',     type=int, default=500)
    p.add_argument('--samplingsteps', type=int, default=500)
    p.add_argument('--objective',     type=str, default='pred_noise')
    p.add_argument('--loss_type',     type=str, default='l2')
    p.add_argument('--beta_schedule', type=str, default='cosine')
    p.add_argument('--cuda_id',       type=str, default='0')
    p.add_argument('--img_dim',       type=int, default=64)
    p.add_argument('--zoom',          type=int, default=15)
    p.add_argument('--emb_file',      type=str, default='')
    p.add_argument('--no_vlm',        action='store_true', help='disable VLM tile embedding')
    p.add_argument('--lr',            type=float, default=1e-3, help='peak learning rate')
    p.add_argument('--exp_name',      type=str, default='', help='experiment name for model save dir')
    args = p.parse_args()
    args.cuda = torch.cuda.is_available()
    args.dim  = 2
    return args

opt = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)
device = torch.device('cuda:0' if opt.cuda else 'cpu')

def mcl_loss(hist_emb, img_emb, temperature=0.1):
    h = F.normalize(hist_emb, dim=-1)
    v = F.normalize(img_emb,  dim=-1)
    sim = torch.mm(h, v.T) / temperature
    labels = torch.arange(sim.shape[0], device=sim.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss

def setup_init():
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    if opt.cuda: torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalization(x, MAX, MIN): return (x - MIN) / (MAX - MIN)

def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y

def load_data():
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', opt.dataset)
    def read(split):
        with open(f'{data_root}/{split}.pkl', 'rb') as f:
            d = pickle.load(f)
        d = [[list(i) for i in u] for u in d]
        d = [[[i[0], i[0]-u[idx-1][0] if idx>0 else i[0]] + i[1:]
              for idx, i in enumerate(u)] for u in d]
        return d
    train_data = read('data_train')
    val_data   = read('data_val')
    test_data  = read('data_test')
    data_all   = train_data + val_data + test_data
    Max, Min = [], []
    for m in range(opt.dim + 2):
        if m > 0:
            Max.append(max(i[m] for u in data_all for i in u))
            Min.append(min(i[m] for u in data_all for i in u))
        else:
            Max.append(1); Min.append(0)
    def norm(d):
        return [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in d]
    return (get_dataloader(norm(train_data), opt.batch_size, D=opt.dim, shuffle=True),
            get_dataloader(norm(val_data),   min(len(val_data), 1000),  D=opt.dim, shuffle=False),
            get_dataloader(norm(test_data),  min(len(test_data), 1000), D=opt.dim, shuffle=False),
            (Max, Min), data_all)

def build_emb_lookup(data_all, emb_cache, zoom):
    lookup = {}
    for seq in data_all:
        for event in seq:
            lon_raw = float(event[2])
            lat_raw = float(event[3])
            tx, ty = lat_lon_to_tile(lat_raw, lon_raw, zoom)
            key = (zoom, tx, ty)
            if key in emb_cache:
                val = emb_cache[key]
                lookup[(round(lon_raw,5), round(lat_raw,5))] = val["embedding"] if isinstance(val, dict) else val
    return lookup

def batch_to_model(batch, transformer, emb_lookup, MAX, MIN):
    event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)
    event_loc = torch.cat((lng.unsqueeze(2), lat.unsqueeze(2)), dim=-1)
    enc_out, mask = transformer(event_loc, event_time_origin)
    B = mask.shape[0]

    enc_out_nm, time_nm, loc_nm, img_nm = [], [], [], []
    zero_emb = torch.zeros(1536)
    for b_idx in range(B):
        length = int(mask[b_idx].sum().item())
        if length <= 1: continue
        lng_raw = (lng[b_idx, :length].cpu() * (MAX[2]-MIN[2])) + MIN[2]
        lat_raw = (lat[b_idx, :length].cpu() * (MAX[3]-MIN[3])) + MIN[3]
        for ev_idx in range(length - 1):
            enc_out_nm.append(enc_out[b_idx][ev_idx].unsqueeze(0))
            time_nm.append(event_time[b_idx][ev_idx+1].unsqueeze(0))
            loc_nm.append(event_loc[b_idx][ev_idx+1].unsqueeze(0))
            lon_k = round(lng_raw[ev_idx+1].item(), 5)
            lat_k = round(lat_raw[ev_idx+1].item(), 5)
            img_nm.append(emb_lookup.get((lon_k, lat_k), zero_emb).unsqueeze(0))

    if len(enc_out_nm) == 0:
        return None, None, None, None
    enc_out_nm = torch.cat(enc_out_nm, dim=0).reshape(-1, 1, enc_out.shape[-1])
    time_nm    = torch.cat(time_nm,    dim=0).reshape(-1, 1, 1)
    loc_nm     = torch.cat(loc_nm,     dim=0).reshape(-1, 1, opt.dim)
    img_nm     = torch.cat(img_nm,     dim=0).unsqueeze(1).to(device)
    return time_nm, loc_nm, enc_out_nm, img_nm

if __name__ == '__main__':
    setup_init()
    setproctitle.setproctitle('MM-STPP')

    emb_file = opt.emb_file or f'tile_embeddings_{opt.dataset}_z{opt.zoom}.pt'
    print(f'Loading tile embeddings from {emb_file}...')
    emb_cache = torch.load(emb_file, map_location='cpu')

    trainloader, valloader, testloader, (MAX, MIN), data_all = load_data()
    emb_lookup = build_emb_lookup(data_all, emb_cache, opt.zoom)

    logdir = f'./logs_mm/{opt.dataset}_ts{opt.timesteps}'
    if opt.exp_name:
        model_path = f'./models_mm/{opt.dataset}_{opt.exp_name}_seed{opt.seed}/'
    else:
        model_path = f'./models_mm/{opt.dataset}_ts{opt.timesteps}_seed{opt.seed}/'
    os.makedirs('./models_mm', exist_ok=True)
    os.makedirs(model_path,   exist_ok=True)
    writer = SummaryWriter(log_dir=logdir, flush_secs=5)

    transformer = Transformer_ST(
        d_model=64, d_rnn=256, d_inner=128,
        n_layers=4, n_head=4, d_k=16, d_v=16,
        dropout=0.1, device=device, loc_dim=opt.dim, CosSin=True
    ).to(device)

    denoiser = ST_Diffusion_MM(
        n_steps=opt.timesteps, dim=1+opt.dim,
        condition=True, cond_dim=64, img_dim=opt.img_dim
    ).to(device)

    diffusion = GaussianDiffusion_MM(
        denoiser, seq_length=1+opt.dim,
        timesteps=opt.timesteps, sampling_timesteps=opt.samplingsteps,
        loss_type=opt.loss_type, objective=opt.objective,
        beta_schedule=opt.beta_schedule
    ).to(device)

    img_projector = ImageProjector(input_dim=1536, img_dim=opt.img_dim).to(device)
    Model = Model_all_MM(
        transformer, diffusion, img_projector,
        n_experts=opt.n_experts, top_k=opt.top_k, knn_k=opt.knn_k,
        use_uncertainty=not opt.wo_uncertainty,
    ).to(device)

    optimizer = AdamW(Model.parameters(), lr=opt.lr, betas=(0.9, 0.99))

    warmup_steps = 5
    step, early_stop, min_loss_val = 0, 0, 1e20

    for itr in range(opt.total_epochs):
        print(f'epoch: {itr}', flush=True)

        if itr % 10 == 0:
            Model.eval()
            with torch.no_grad():
                loss_val, mae_spatial, total_num = 0., 0., 0
                for batch in valloader:
                    t_nm, l_nm, enc_nm, img_nm = batch_to_model(
                        batch, Model.transformer, emb_lookup, MAX, MIN)
                    if t_nm is None: continue
                    if opt.no_vlm:
                        img_proj = torch.zeros(img_nm.shape[0], 1, opt.img_dim, device=device)
                    else:
                        img_proj = Model.img_projector(img_nm)
                    # UGMoE fusion at inference
                    hist_emb = enc_nm[:, 0, :64].to(device)
                    spat_emb = enc_nm[:, 0, 64:128].to(device)
                    img_emb  = img_proj[:, 0, :].to(device)
                    mu_t, sig_t = Model.temporal_gauss(hist_emb)
                    mu_s, sig_s = Model.spatial_gauss(spat_emb)
                    mu_i, sig_i = Model.image_gauss(img_emb)
                    mu_fused, _, _ = Model.ugmoe(mu_t, mu_s, mu_i, sig_t, sig_s, sig_i)
                    img_cond_fused = mu_fused.unsqueeze(1)
                    x_input = torch.cat((t_nm, l_nm), dim=-1)
                    loss = Model.diffusion(x_input, enc_nm, img_cond=img_cond_fused)
                    loss_val += loss.item() * t_nm.shape[0]
                    sampled = Model.diffusion.sample(
                        batch_size=t_nm.shape[0], cond=enc_nm, img_cond=img_cond_fused)
                    real = (l_nm[:,0,:2].cpu() + torch.tensor([MIN[2:]])) * \
                           (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]]))
                    gen  = (sampled[:,0,1:3].cpu() + torch.tensor([MIN[2:]])) * \
                           (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]]))
                    mae_spatial += torch.sqrt(((real-gen)**2).sum(dim=-1)).sum().item()
                    total_num   += t_nm.shape[0]
                if total_num > 0:
                    lv = loss_val / total_num
                    ds = mae_spatial / total_num
                    print(f'  [val] loss={lv:.4f}  dist_spatial={ds:.4f}', flush=True)
                    writer.add_scalar('Eval/loss_val',         lv, itr)
                    writer.add_scalar('Eval/distance_spatial', ds, itr)
                    if loss_val > min_loss_val:
                        early_stop += 1
                        if early_stop >= 100:
                            print('Early stopping.'); break
                    else:
                        early_stop = 0
                        torch.save(Model.state_dict(), model_path + f'model_{itr}.pkl')
                        torch.save(Model.state_dict(), model_path + 'model_best.pkl')
                    min_loss_val = min(min_loss_val, loss_val)

        min_lr  = opt.lr * 0.05
        base_lr = opt.lr*(itr+1)/warmup_steps if itr < warmup_steps \
                   else opt.lr - (opt.lr-min_lr)*(itr-warmup_steps)/opt.total_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr
        writer.add_scalar('Stats/lr', base_lr, itr)

        Model.train()
        loss_epoch, total_num = 0., 0
        for batch in trainloader:
            t_nm, l_nm, enc_nm, img_nm = batch_to_model(
                batch, Model.transformer, emb_lookup, MAX, MIN)
            if t_nm is None: continue

            if opt.no_vlm:
                img_proj = torch.zeros(img_nm.shape[0], 1, opt.img_dim, device=device)
            else:
                img_proj = Model.img_projector(img_nm)

            # ── event-level embeddings ──────────────────────────────────────
            hist_emb = enc_nm[:, 0, :64].to(device)      # [N, 64] temporal
            spat_emb = enc_nm[:, 0, 64:128].to(device)   # [N, 64] spatial
            img_emb  = img_proj[:, 0, :].to(device)      # [N, 64] image

            # ── Gaussian uncertainty heads ──────────────────────────────────
            mu_t, sig_t = Model.temporal_gauss(hist_emb)  # [N, 64] each
            mu_s, sig_s = Model.spatial_gauss(spat_emb)   # [N, 64] each
            mu_i, sig_i = Model.image_gauss(img_emb)      # [N, 64] each

            # ── MutualKNN alignment loss ────────────────────────────────────
            loss_knn = Model.mutual_knn(mu_t, mu_s, mu_i, sig_i)

            # ── UGMoE fusion ────────────────────────────────────────────────
            z_t = mu_t + sig_t * torch.randn_like(sig_t)
            z_s = mu_s + sig_s * torch.randn_like(sig_s)
            z_i = mu_i + sig_i * torch.randn_like(sig_i)
            mu_fused, _, loss_balance = Model.ugmoe(z_t, z_s, z_i, sig_t, sig_s, sig_i)

            # Use UGMoE output as image conditioning to diffusion
            img_cond_fused = mu_fused.unsqueeze(1)   # [N, 1, 64]

            x_input   = torch.cat((t_nm, l_nm), dim=-1)
            loss_diff = Model.diffusion(x_input, enc_nm, img_cond=img_cond_fused)

            # ── MCL on fused representation ─────────────────────────────────
            hist_for_mcl = Model.hist_proj(mu_fused)
            img_for_mcl  = Model.img_proj_mcl(img_emb)
            loss_mcl = mcl_loss(hist_for_mcl, img_for_mcl)

            loss = (loss_diff
                    + opt.alpha     * loss_mcl
                    + opt.alpha_knn * loss_knn
                    + opt.alpha_bal * loss_balance)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step()

            loss_epoch += loss.item() * t_nm.shape[0]
            total_num  += t_nm.shape[0]
            writer.add_scalar('Train/loss_step',    loss.item(),          step)
            writer.add_scalar('Train/loss_diff',    loss_diff.item(),     step)
            writer.add_scalar('Train/loss_mcl',     loss_mcl.item(),      step)
            writer.add_scalar('Train/loss_knn',     loss_knn.item(),      step)
            writer.add_scalar('Train/loss_balance', loss_balance.item(),  step)
            step += 1

        writer.add_scalar('Train/loss_epoch', loss_epoch / max(total_num, 1), itr)
        if opt.cuda:
            torch.cuda.empty_cache()
