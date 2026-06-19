"""
Extract Qwen2-VL tile embeddings for GOES-18 OSM tiles (zoom=7).
Run after: crawl_tiles_goes18.py
Output: ~/MultiModal_STPP/tile_embeddings_GOES18_z7_flat.pt
        Format: {(7, tx, ty): tensor[1536]}

Run: conda run -n stdpp python extract_tile_embeddings_goes18.py
"""
import os, math, torch
from pathlib import Path
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

ZOOM     = 7
LON_MIN, LON_MAX = -122, -116
LAT_MIN, LAT_MAX = 31, 36
TILE_DIR = "osm_tiles_GOES18_z7"
OUT_FILE = "tile_embeddings_GOES18_z7_flat.pt"
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE   = "cuda:0"

def lon_to_x(lon, zoom): return int((lon + 180) / 360 * (2 ** zoom))
def lat_to_y(lat, zoom):
    lat_rad = math.radians(lat)
    return int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * (2 ** zoom))

x_min = lon_to_x(LON_MIN, ZOOM)
x_max = lon_to_x(LON_MAX, ZOOM)
y_min = lat_to_y(LAT_MAX, ZOOM)
y_max = lat_to_y(LAT_MIN, ZOOM)

print(f"Loading Qwen2-VL...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)

embeddings = {}
tiles = [(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
total = len(tiles)
print(f"Processing {total} tiles (x=[{x_min},{x_max}], y=[{y_min},{y_max}])")

for i, (x, y) in enumerate(tiles):
    tile_path = Path(TILE_DIR) / str(ZOOM) / str(x) / f"{y}.png"
    if not tile_path.exists():
        print(f"  [{i+1}/{total}] MISSING: {tile_path}")
        continue
    image = Image.open(tile_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": "Describe this map tile."}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        emb = out.hidden_states[-1][0, -1, :].float().cpu()
    embeddings[(ZOOM, x, y)] = emb
    print(f"  [{i+1}/{total}] ({ZOOM},{x},{y}) shape={emb.shape}")

torch.save(embeddings, OUT_FILE)
print(f"\nSaved {len(embeddings)} embeddings → {OUT_FILE}")
