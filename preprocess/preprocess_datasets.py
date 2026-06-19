"""
Preprocess all satellite/airborne fire/smoke label TIF files into
DSTPP-format pickle datasets (list of sequences of [time_hours, lon, lat]).

Datasets produced:
  - GK2A_fire      (from DSTPP/dataset/GK2A/Fire/)
  - eMAS_smoke     (from raw_labels/eMAS/Smoke/)
  - AirMSPI_smoke  (from raw_labels/AirMSPI_Sweep/Smoke/)
  - MASTER_fire    (from raw_labels/MASTER/Fire/)

Usage:
    python preprocess_datasets.py
"""

import rasterio
import numpy as np
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta

BASE     = Path('.')                    # project root
OUT_BASE = Path('data')                 # output dataset directory
RAW      = Path('raw_data')             # path to raw label .tif files
GK2A_FIRE_DIR = RAW / 'GK2A' / 'Fire'  # path to GK2A fire .tif files

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15


def save_splits(sequences, out_dir, seq_name, max_seqs=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_seqs and len(sequences) > max_seqs:
        sequences = sequences[:max_seqs]
        print(f"  [{seq_name}] Truncated to first {max_seqs} sequences (time-based)")
    n = len(sequences)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits = {
        'data_train': sequences[:n_train],
        'data_val':   sequences[n_train:n_train + n_val],
        'data_test':  sequences[n_train + n_val:],
    }
    for name, data in splits.items():
        path = out_dir / f'{name}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    t_start = sequences[0][0][0]
    t_end   = sequences[-1][-1][0]
    print(f"  [{seq_name}] Train:{len(splits['data_train'])}  Val:{len(splits['data_val'])}  Test:{len(splits['data_test'])}  (t={t_start:.1f}h ~ {t_end:.1f}h)")


def make_sequences(all_events, seq_len):
    all_events.sort(key=lambda x: x[0])
    seqs = []
    for i in range(0, len(all_events) - seq_len + 1, seq_len):
        seqs.append(all_events[i:i + seq_len])
    return seqs


def read_tif_events(f, t_start_h, t_end_h, grid):
    """Extract (t, lon, lat) events from a label tif using uniform time spread."""
    with rasterio.open(f) as src:
        arr = src.read(1)
        tf  = src.transform
        ys, xs = np.where(arr > 0)
        if len(xs) == 0:
            return []
        lons = tf.c + xs * tf.a
        lats = tf.f + ys * tf.e

    gx = np.round(lons / grid).astype(int)
    gy = np.round(lats / grid).astype(int)
    unique = np.unique(np.stack([gx, gy], axis=1), axis=0)
    cell_lons = unique[:, 0] * grid
    cell_lats = unique[:, 1] * grid

    order = np.argsort(-cell_lats)
    cell_lons = cell_lons[order]
    cell_lats = cell_lats[order]

    n = len(cell_lons)
    duration = max(t_end_h - t_start_h, 1/60.0)
    offsets  = np.linspace(0, duration, n, endpoint=False)
    times    = t_start_h + offsets

    return [[float(t), float(lon), float(lat)]
            for t, lon, lat in zip(times, cell_lons, cell_lats)]


# ──────────────────────────────────────────────
# 1. GK2A Fire  (10 min intervals, geostationary)
# ──────────────────────────────────────────────
print("\n=== GK2A Fire ===")
files = sorted(f for f in GK2A_FIRE_DIR.glob('*.Fire.tif') if not f.name.startswith('._'))
print(f"Found {len(files)} tif files")
t0, all_events = None, []
for f in files:
    m = re.search(r'(\d{12})', f.name)
    if not m:
        continue
    dt = datetime.strptime(m.group(1), '%Y%m%d%H%M')
    if t0 is None:
        t0 = dt
    t_h = (dt - t0).total_seconds() / 3600.0
    events = read_tif_events(f, t_h, t_h + 10/60.0, grid=0.05)
    all_events.extend(events)
    print(f"  {dt:%Y-%m-%d %H:%M} (t={t_h:.1f}h): {len(events)} cells")

seqs = make_sequences(all_events, seq_len=10)
print(f"Total events: {len(all_events)} → {len(seqs)} sequences")
save_splits(seqs, OUT_BASE / 'GK2A_fire', 'GK2A_fire')


# ──────────────────────────────────────────────
# 2. eMAS Smoke  (airborne overpass)
# ──────────────────────────────────────────────
print("\n=== eMAS Smoke ===")
smoke_dir = RAW / 'eMAS' / 'Smoke'
files = sorted(f for f in smoke_dir.glob('*.smoke.tif') if not f.name.startswith('._'))
print(f"Found {len(files)} tif files")
t0, all_events = None, []
for f in files:
    m = re.search(r'_(\d{8})_(\d{4})_(\d{4})_', f.name)
    if not m:
        continue
    dt_start = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M")
    dt_end   = datetime.strptime(f"{m.group(1)}{m.group(3)}", "%Y%m%d%H%M")
    if dt_end < dt_start:
        dt_end += timedelta(days=1)
    if t0 is None:
        t0 = dt_start
    t_s = (dt_start - t0).total_seconds() / 3600.0
    t_e = (dt_end   - t0).total_seconds() / 3600.0
    events = read_tif_events(f, t_s, t_e, grid=0.001)
    all_events.extend(events)
    print(f"  {m.group(1)} {m.group(2)} (t={t_s:.1f}h): {len(events)} cells")

seqs = make_sequences(all_events, seq_len=50)
print(f"Total events: {len(all_events)} → {len(seqs)} sequences")
save_splits(seqs, OUT_BASE / 'eMAS_smoke', 'eMAS_smoke', max_seqs=5700)


# ──────────────────────────────────────────────
# 3. AirMSPI Smoke  (airborne sweep)
# ──────────────────────────────────────────────
print("\n=== AirMSPI Smoke ===")
smoke_dir = RAW / 'AirMSPI_Sweep' / 'Smoke'
files = sorted(f for f in smoke_dir.glob('*.smoke.tif') if not f.name.startswith('._'))
print(f"Found {len(files)} tif files")
t0, all_events = None, []
for f in files:
    # AirMSPI_ER2_GRP_TERRAIN_20190806_181534Z_...
    m = re.search(r'_(\d{8})_(\d{6})Z_', f.name)
    if not m:
        continue
    dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
    if t0 is None:
        t0 = dt
    t_h = (dt - t0).total_seconds() / 3600.0
    # Each sweep takes ~1 minute
    events = read_tif_events(f, t_h, t_h + 1/60.0, grid=0.001)
    all_events.extend(events)
    print(f"  {dt:%Y-%m-%d %H:%M:%S} (t={t_h:.2f}h): {len(events)} cells")

seqs = make_sequences(all_events, seq_len=50)
print(f"Total events: {len(all_events)} → {len(seqs)} sequences")
save_splits(seqs, OUT_BASE / 'AirMSPI_smoke', 'AirMSPI_smoke', max_seqs=1430)


# ──────────────────────────────────────────────
# 4. MASTER Fire  (airborne overpass)
# ──────────────────────────────────────────────
print("\n=== MASTER Fire ===")
fire_dir = RAW / 'MASTER' / 'Fire'
files = sorted(f for f in fire_dir.glob('*.fire.tif') if not f.name.startswith('._'))
print(f"Found {len(files)} tif files")
t0, all_events = None, []
for f in files:
    # MASTERL1B_1981719_02_20190806_1851_1900_V01.fire.tif
    m = re.search(r'_(\d{8})_(\d{4})_(\d{4})_', f.name)
    if not m:
        continue
    dt_start = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M")
    dt_end   = datetime.strptime(f"{m.group(1)}{m.group(3)}", "%Y%m%d%H%M")
    if dt_end < dt_start:
        dt_end += timedelta(days=1)
    if t0 is None:
        t0 = dt_start
    t_s = (dt_start - t0).total_seconds() / 3600.0
    t_e = (dt_end   - t0).total_seconds() / 3600.0
    events = read_tif_events(f, t_s, t_e, grid=0.001)
    all_events.extend(events)
    print(f"  {m.group(1)} {m.group(2)} (t={t_s:.1f}h): {len(events)} cells")

seqs = make_sequences(all_events, seq_len=50)
print(f"Total events: {len(all_events)} → {len(seqs)} sequences")
save_splits(seqs, OUT_BASE / 'MASTER_fire', 'MASTER_fire')

print("\n=== All done ===")
print("Datasets in", OUT_BASE)
for d in sorted(OUT_BASE.iterdir()):
    if d.is_dir():
        sizes = []
        for split in ['data_train','data_val','data_test']:
            p = d / f'{split}.pkl'
            if p.exists():
                import pickle as pk
                with open(p,'rb') as f2:
                    sizes.append(len(pk.load(f2)))
        print(f"  {d.name}: {sizes}")
