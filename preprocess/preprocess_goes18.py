"""
Preprocess GOES-18 fire and smoke label TIF files into DSTPP-format datasets.

Datasets produced:
  - GOES18_fire   (from raw_labels/GOES-18/Fire/)
  - GOES18_smoke  (from raw_labels/GOES-18/Smoke/)

GOES-18 filename format:
  OR_ABI-L1b-RadC-M6C01_G18_s{YYYY}{DDD}{HH}{MM}{sss}_e..._c....{fire|smoke}.tif
  where DDD = day-of-year, sss = tenths-of-second

Usage:
    conda run -n stdpp python preprocess_goes18.py
"""
import rasterio
import numpy as np
import pickle
import re
from pathlib import Path
from datetime import datetime, timedelta

BASE     = Path('.')                      # project root
RAW      = Path('raw_data') / 'GOES-18'  # path to GOES-18 fire/smoke .tif files

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
GRID = 0.02  # ~2km, same as GOES-17 smoke


def save_splits(sequences, out_dir, name, max_seqs=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if max_seqs and len(sequences) > max_seqs:
        sequences = sequences[:max_seqs]
        print(f"  [{name}] Truncated to {max_seqs} sequences")
    n = len(sequences)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    splits = {
        'data_train': sequences[:n_train],
        'data_val':   sequences[n_train:n_train + n_val],
        'data_test':  sequences[n_train + n_val:],
    }
    for sname, data in splits.items():
        with open(out_dir / f'{sname}.pkl', 'wb') as f:
            pickle.dump(data, f)
    print(f"  [{name}] train:{len(splits['data_train'])}  val:{len(splits['data_val'])}  test:{len(splits['data_test'])}")


def make_sequences(all_events, seq_len):
    all_events.sort(key=lambda x: x[0])
    seqs = []
    for i in range(0, len(all_events) - seq_len + 1, seq_len):
        seqs.append(all_events[i:i + seq_len])
    return seqs


def parse_goes18_time(fname):
    # s{YYYY}{DDD}{HH}{MM}{sss}
    m = re.search(r'_s(\d{4})(\d{3})(\d{2})(\d{2})\d{3}_', fname)
    if not m:
        return None
    year, doy, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm)


def read_tif_events(f, t_start_h, t_end_h):
    with rasterio.open(f) as src:
        arr = src.read(1)
        tf  = src.transform
        ys, xs = np.where(arr > 0)
        if len(xs) == 0:
            return []
        lons = tf.c + xs * tf.a
        lats = tf.f + ys * tf.e

    gx = np.round(lons / GRID).astype(int)
    gy = np.round(lats / GRID).astype(int)
    unique = np.unique(np.stack([gx, gy], axis=1), axis=0)
    cell_lons = unique[:, 0] * GRID
    cell_lats = unique[:, 1] * GRID

    order = np.argsort(-cell_lats)
    cell_lons = cell_lons[order]
    cell_lats = cell_lats[order]

    n = len(cell_lons)
    duration = max(t_end_h - t_start_h, 1 / 60.0)
    offsets  = np.linspace(0, duration, n, endpoint=False)
    times    = t_start_h + offsets

    return [[float(t), float(lon), float(lat)]
            for t, lon, lat in zip(times, cell_lons, cell_lats)]


for label_type, ext, seq_len, max_seqs in [
    ('Fire',  'fire',  10, None),
    ('Smoke', 'smoke', 20, 5700),
]:
    print(f"\n=== GOES-18 {label_type} ===")
    label_dir = RAW / label_type
    files = sorted(f for f in label_dir.glob(f'*.{ext}.tif') if not f.name.startswith('._'))
    print(f"Found {len(files)} tif files")

    t0, all_events = None, []
    for f in files:
        dt = parse_goes18_time(f.name)
        if dt is None:
            print(f"  WARNING: could not parse time from {f.name}")
            continue
        if t0 is None:
            t0 = dt
        t_h = (dt - t0).total_seconds() / 3600.0
        events = read_tif_events(f, t_h, t_h + 1.0)  # 1-hour observation window
        all_events.extend(events)
        print(f"  {dt:%Y-%m-%d %H:%M} (t={t_h:.1f}h): {len(events)} cells")

    seqs = make_sequences(all_events, seq_len)
    print(f"Total events: {len(all_events)} → {len(seqs)} sequences (seq_len={seq_len})")

    ds_name = f'GOES18_{label_type.lower()}'
    save_splits(seqs, Path('data') / ds_name, ds_name, max_seqs=max_seqs)

print("\n=== Done ===")
