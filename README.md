# Uncertainty-aware Multi-modality Spatio-temporal Wildfire and Smoke Prediction

Code for the SIGSPATIAL 2026 submission.

## Requirements

```bash
pip install -r requirements.txt
```

## Data

Each dataset must be placed at `data/{DATASET}/` and contain three pickle files:

```
data/
  eMAS/          data_train.pkl  data_val.pkl  data_test.pkl
  GK2A_fire/     ...
  eMAS_smoke/    ...
  GOES18_smoke/  ...
```

Each pickle file is a `list` of sequences. Each sequence is a `list` of events `[time_hours, lon, lat]`.

Preprocessing scripts for each dataset are provided in `preprocess/`. Update the `raw_data/` paths at the top of each script to point to your local raw TIF files before running.

## Tile Embeddings

Satellite tile embeddings are extracted using Qwen2-VL-2B and OpenStreetMap tiles:

```bash
# 1. Download OSM tiles for the region of interest
python extract_embeddings/crawl_tiles_eMAS.py
python extract_embeddings/crawl_tiles_goes18.py

# 2. Extract Qwen2-VL embeddings
python extract_embeddings/extract_tile_embeddings_eMAS.py
python extract_embeddings/extract_tile_embeddings_goes18.py
```

This produces `tile_embeddings_*.pt` files in the project root.

## Training and Evaluation

```bash
# Train and evaluate on all four datasets (GPU 0, seed 42)
bash run.sh 0 42
```

To run a single dataset manually:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset eMAS \
    --zoom 7 \
    --emb_file tile_embeddings_eMAS_z7_flat.pt \
    --total_epochs 500 \
    --seed 42 \
    --alpha_knn 0.005

CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataset eMAS \
    --zoom 7 \
    --emb_file tile_embeddings_eMAS_z7_flat.pt \
    --model_ckpt models_mm/eMAS_seed42/model_best.pkl \
    --n_samples 3
```

## Ablation

Key flags for ablation variants:

| Variant | Flag |
|---|---|
| w/o MutualKNN | `--alpha_knn 0` |
| w/o Uncertainty | `--wo_uncertainty` |
| w/o Satellite | `--no_vlm --alpha_knn 0` |

## Project Structure

```
UMS/
├── model_mm.py            # UMS model (UGMoE + ST-Diffusion)
├── train.py               # Training script
├── test.py                # Evaluation script
├── run.sh                 # End-to-end train+eval for all datasets
├── requirements.txt
├── data/                  # Preprocessed datasets (place here)
├── preprocess/            # Dataset preprocessing scripts
│   ├── preprocess_eMAS.py
│   ├── preprocess_datasets.py   (GK2A_fire, eMAS_smoke)
│   └── preprocess_goes18.py
└── extract_embeddings/    # OSM tile crawling + Qwen2-VL embedding
    ├── crawl_tiles_eMAS.py
    ├── crawl_tiles_goes18.py
    ├── extract_tile_embeddings_eMAS.py
    └── extract_tile_embeddings_goes18.py
```
