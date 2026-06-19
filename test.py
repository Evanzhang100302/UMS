import sys, os, argparse, math, pickle, random
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--dataset",    type=str,  default="Earthquake")
p.add_argument("--zoom",       type=int,  default=12)
p.add_argument("--timesteps",  type=int,  default=500)
p.add_argument("--samplingsteps", type=int, default=500)
p.add_argument("--batch_size", type=int,  default=64)
p.add_argument("--cuda_id",    type=str,  default="0")
p.add_argument("--model_ckpt", type=str,  default="")
p.add_argument("--emb_file",   type=str,  default="")
p.add_argument("--n_samples",  type=int,  default=3)
p.add_argument("--seed",       type=int,  default=42)
p.add_argument("--img_dim",    type=int,  default=64)
p.add_argument("--no_vlm",     action="store_true")
args = p.parse_args()

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder import Transformer_ST as _Transformer_ST_Base, get_dataloader

class Transformer_ST(_Transformer_ST_Base):
    def forward(self, event_loc, event_time):
        enc_output_all, non_pad_mask = super().forward(event_loc, event_time)
        return enc_output_all[:, :, :128], non_pad_mask
from model_mm import Model_all_MM, GaussianDiffusion_MM, ST_Diffusion_MM, ImageProjector

def normalization(x, MAX, MIN): return (x - MIN) / (MAX - MIN)

def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y

def load_data():
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", args.dataset)
    def read(split):
        with open(f"{data_root}/{split}.pkl", "rb") as f:
            d = pickle.load(f)
        d = [[list(i) for i in u] for u in d]
        d = [[[i[0], i[0]-u[idx-1][0] if idx>0 else i[0]] + i[1:]
              for idx, i in enumerate(u)] for u in d]
        return d
    train_data = read("data_train")
    val_data   = read("data_val")
    test_data  = read("data_test")
    data_all   = train_data + val_data + test_data
    dim = 2
    Max, Min = [], []
    for m in range(dim + 2):
        if m > 0:
            Max.append(max(i[m] for u in data_all for i in u))
            Min.append(min(i[m] for u in data_all for i in u))
        else:
            Max.append(1); Min.append(0)
    def norm(d):
        return [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in d]
    return (get_dataloader(norm(test_data), min(len(test_data), 1000), D=dim, shuffle=False),
            (Max, Min), data_all)

def build_emb_lookup(data_all, emb_cache):
    lookup = {}
    for seq in data_all:
        for event in seq:
            lon_raw = float(event[2]); lat_raw = float(event[3])
            tx, ty = lat_lon_to_tile(lat_raw, lon_raw, args.zoom)
            key = (args.zoom, tx, ty)
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
    loc_nm     = torch.cat(loc_nm,     dim=0).reshape(-1, 1, 2)
    img_nm     = torch.cat(img_nm,     dim=0).unsqueeze(1).to(device)
    return time_nm, loc_nm, enc_out_nm, img_nm

# ── load data & embeddings ────────────────────────────────────────────────────

emb_file = args.emb_file or f"tile_embeddings_{args.dataset}_z{args.zoom}.pt"

testloader, (MAX, MIN), data_all = load_data()
emb_cache  = torch.load(emb_file, map_location="cpu")
emb_lookup = build_emb_lookup(data_all, emb_cache)

# ── build model ───────────────────────────────────────────────────────────────

transformer = Transformer_ST(
    d_model=64, d_rnn=256, d_inner=128,
    n_layers=4, n_head=4, d_k=16, d_v=16,
    dropout=0.1, device=device, loc_dim=2, CosSin=True
).to(device)
denoiser  = ST_Diffusion_MM(n_steps=args.timesteps, dim=3, condition=True, cond_dim=64, img_dim=args.img_dim).to(device)
diffusion = GaussianDiffusion_MM(
    denoiser, seq_length=3,
    timesteps=args.timesteps, sampling_timesteps=args.samplingsteps,
    loss_type="l2", objective="pred_noise", beta_schedule="cosine"
).to(device)
img_projector = ImageProjector(input_dim=1536, img_dim=args.img_dim).to(device)
Model = Model_all_MM(transformer, diffusion, img_projector).to(device)

# ── load checkpoint ───────────────────────────────────────────────────────────

Model.load_state_dict(torch.load(args.model_ckpt, map_location=device))
Model.eval()
print(f"Loaded model from {args.model_ckpt}")

# ── evaluation ────────────────────────────────────────────────────────────────

mae_spatial, mae_temporal, rmse_temporal, total_num = 0., 0., 0., 0
with torch.no_grad():
    for batch in testloader:
        t_nm, loc_nm, enc_nm, img_nm = batch_to_model(
            batch, Model.transformer, emb_lookup, MAX, MIN)
        if t_nm is None: continue

        if args.no_vlm:
            img_proj = torch.zeros(img_nm.shape[0], 1, args.img_dim, device=device)
        else:
            img_proj = Model.img_projector(img_nm)

        hist_emb = enc_nm[:, 0, :64].to(device)
        spat_emb = enc_nm[:, 0, 64:128].to(device)
        img_emb  = img_proj[:, 0, :].to(device)
        mu_t, sig_t = Model.temporal_gauss(hist_emb)
        mu_s, sig_s = Model.spatial_gauss(spat_emb)
        mu_i, sig_i = Model.image_gauss(img_emb)
        mu_fused, _, _ = Model.ugmoe(mu_t, mu_s, mu_i, sig_t, sig_s, sig_i)
        img_cond_fused = mu_fused.unsqueeze(1)

        gen_t_all, gen_s_all = [], []
        for _ in range(args.n_samples):
            sampled = Model.diffusion.sample(batch_size=t_nm.shape[0], cond=enc_nm, img_cond=img_cond_fused)
            gen_t_all.append(sampled[:, 0, :1].cpu())
            gen_s_all.append(sampled[:, 0, -2:].cpu())
        gen_t = torch.stack(gen_t_all, dim=0).mean(dim=0)
        gen_s = torch.stack(gen_s_all, dim=0).mean(dim=0)

        real_t = (t_nm[:, 0, :].cpu() * (MAX[1] - MIN[1])) + MIN[1]
        gen_t  = (gen_t * (MAX[1] - MIN[1])) + MIN[1]
        mae_temporal  += torch.abs(real_t - gen_t).sum().item()
        rmse_temporal += ((real_t - gen_t) ** 2).sum().item()

        real_s = (loc_nm[:, 0, :].cpu() * torch.tensor([MAX[2]-MIN[2], MAX[3]-MIN[3]])) + torch.tensor([MIN[2], MIN[3]])
        gen_s  = (gen_s * torch.tensor([MAX[2]-MIN[2], MAX[3]-MIN[3]])) + torch.tensor([MIN[2], MIN[3]])
        mae_spatial += torch.sqrt(((real_s - gen_s) ** 2).sum(dim=-1)).sum().item()
        total_num   += t_nm.shape[0]

print(f"Test events   : {total_num}")
print(f"MAE  temporal : {mae_temporal/total_num:.4f}")
print(f"RMSE temporal : {(rmse_temporal/total_num)**0.5:.4f}")
print(f"MAE  spatial  : {mae_spatial/total_num:.4f}")
