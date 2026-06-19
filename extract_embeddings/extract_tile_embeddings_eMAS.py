"""
Extract Qwen2-VL tile embeddings for eMAS OSM tiles (zoom=7).
Run on server: python extract_tile_embeddings_eMAS.py
Requires: conda activate stdpp, GPU available, osm_tiles_eMAS_z7/ directory
"""
import os
import math
import torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ---- Config ----
ZOOM = 7
LON_MIN, LON_MAX = -120, -112
LAT_MIN, LAT_MAX = 34, 48
TILE_DIR = "osm_tiles_eMAS_z7"
OUTPUT_FILE = "tile_embeddings_eMAS_z7.pt"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda:0"
# ----------------

def lon_to_x(lon, zoom):
    return int((lon + 180) / 360 * (2 ** zoom))

def lat_to_y(lat, zoom):
    lat_rad = math.radians(lat)
    return int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * (2 ** zoom))

def tile_center(z, x, y):
    n = 2 ** z
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat

x_min = lon_to_x(LON_MIN, ZOOM)
x_max = lon_to_x(LON_MAX, ZOOM)
y_min = lat_to_y(LAT_MAX, ZOOM)
y_max = lat_to_y(LAT_MIN, ZOOM)

print(f"Loading Qwen2-VL model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16
).to(DEVICE)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print(f"Model loaded. Processing {(x_max-x_min+1)*(y_max-y_min+1)} tiles...")

embeddings = {}
total = (x_max - x_min + 1) * (y_max - y_min + 1)
count = 0

for x in range(x_min, x_max + 1):
    for y in range(y_min, y_max + 1):
        tile_path = Path(TILE_DIR) / str(ZOOM) / str(x) / f"{y}.png"
        if not tile_path.exists():
            print(f"  MISSING: {tile_path}")
            continue

        image = Image.open(tile_path).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this map tile."}
        ]}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            emb = output.hidden_states[-1][0, -1, :].float().cpu()

        lon, lat = tile_center(ZOOM, x, y)
        key = (ZOOM, x, y)
        embeddings[key] = {
            "embedding": emb,
            "lon": lon,
            "lat": lat
        }

        count += 1
        print(f"  [{count}/{total}] tile ({ZOOM},{x},{y}): lon={lon:.2f}, lat={lat:.2f}, emb shape={emb.shape}")

torch.save(embeddings, OUTPUT_FILE)
print(f"\nSaved {len(embeddings)} embeddings to {OUTPUT_FILE}")
print(f"Embedding dim: {list(embeddings.values())[0]['embedding'].shape}")
