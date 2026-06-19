"""
Preprocess eMAS fire TIF files into DSTPP-format pickle datasets.

Strategy:
- Downsample each overpass to 0.001-degree (~100m) grid cells
- Assign time within overpass based on spatial scan order (lat desc)
- Sort all events by time, create sequences of seq_len=50
- 70/15/15 train/val/test split

Output format: list of sequences, each sequence = list of [time_hours, lon, lat]

Usage:
    python preprocess_eMAS.py
"""

import rasterio
import numpy as np
import pickle
import re
from pathlib import Path
from datetime import datetime

# ---- Config ----
FIRE_DIR = Path('raw_data/eMAS/Fire')   # path to eMAS Level-1B fire .tif files
OUT_DIR  = Path('data/eMAS')            # output dataset directory
GRID     = 0.001   # degrees (~100m)
SEQ_LEN  = 50
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# ----------------

OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(f for f in FIRE_DIR.glob('*.fire.tif') if not f.name.startswith('._'))
print(f"Found {len(files)} fire TIF files")

# Reference time: start of first overpass
t0 = None
all_events = []  # list of [t_hours, lon, lat]

for f in files:
    # Parse filename: eMASL1B_19910_06_20190806_1815_1824_V03.fire.tif
    m = re.search(r'_(\d{8})_(\d{4})_(\d{4})_', f.name)
    if not m:
        print(f"  Skipping (can't parse time): {f.name}")
        continue

    date_str, t_start_str, t_end_str = m.group(1), m.group(2), m.group(3)
    dt_start = datetime.strptime(f"{date_str}{t_start_str}", "%Y%m%d%H%M")
    dt_end   = datetime.strptime(f"{date_str}{t_end_str}",   "%Y%m%d%H%M")

    # Handle overnight flights (end time < start time → next day)
    if dt_end < dt_start:
        from datetime import timedelta
        dt_end += timedelta(days=1)

    if t0 is None:
        t0 = dt_start

    t_start_h = (dt_start - t0).total_seconds() / 3600.0
    t_end_h   = (dt_end   - t0).total_seconds() / 3600.0
    duration_h = max(t_end_h - t_start_h, 1/60.0)  # at least 1 minute

    with rasterio.open(f) as src:
        arr       = src.read(1)
        transform = src.transform
        ys, xs    = np.where(arr > 0)
        if len(xs) == 0:
            print(f"  {f.name}: no fire pixels")
            continue

        lons = transform.c + xs * transform.a  # raw lon per pixel
        lats = transform.f + ys * transform.e  # raw lat per pixel

    # Downsample to GRID cells (take cell center)
    grid_x = np.round(lons / GRID).astype(int)
    grid_y = np.round(lats / GRID).astype(int)
    unique_cells = np.unique(np.stack([grid_x, grid_y], axis=1), axis=0)
    cell_lons = unique_cells[:, 0] * GRID
    cell_lats = unique_cells[:, 1] * GRID

    # Assign time within overpass: sort by lat descending (aircraft scans N→S)
    order = np.argsort(-cell_lats)  # descending lat = earlier in scan
    cell_lons = cell_lons[order]
    cell_lats = cell_lats[order]

    n = len(cell_lons)
    # Spread times uniformly across overpass duration
    t_offsets = np.linspace(0, duration_h, n, endpoint=False)
    cell_times = t_start_h + t_offsets

    for t, lon, lat in zip(cell_times, cell_lons, cell_lats):
        all_events.append([float(t), float(lon), float(lat)])

    print(f"  {date_str} {t_start_str} (t={t_start_h:.1f}h): {len(xs)} pixels → {n} grid cells")

# Sort all events by time
all_events.sort(key=lambda x: x[0])
print(f"\nTotal events: {len(all_events)}")
print(f"Time range: {all_events[0][0]:.2f}h ~ {all_events[-1][0]:.2f}h")
print(f"Lon range:  {min(e[1] for e in all_events):.4f} ~ {max(e[1] for e in all_events):.4f}")
print(f"Lat range:  {min(e[2] for e in all_events):.4f} ~ {max(e[2] for e in all_events):.4f}")

# Create sequences of SEQ_LEN events
sequences = []
for i in range(0, len(all_events) - SEQ_LEN + 1, SEQ_LEN):
    seq = all_events[i : i + SEQ_LEN]
    sequences.append(seq)

print(f"\nTotal sequences (len={SEQ_LEN}): {len(sequences)}")

# Train / val / test split
n_train = int(len(sequences) * TRAIN_RATIO)
n_val   = int(len(sequences) * VAL_RATIO)
train = sequences[:n_train]
val   = sequences[n_train : n_train + n_val]
test  = sequences[n_train + n_val :]

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Save
for name, data in [('data_train', train), ('data_val', val), ('data_test', test)]:
    path = OUT_DIR / f'{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved: {path}")

print("\nDone.")
