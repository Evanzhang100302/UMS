"""
Crawl OSM tiles for the eMAS fire region.
eMAS dataset: lon -120~-112, lat 34~48
Zoom 7: ~32 tiles

Run on server: python crawl_tiles_eMAS.py
"""
import os
import math
import time
import requests
from pathlib import Path

ZOOM = 7
LON_MIN, LON_MAX = -120, -112
LAT_MIN, LAT_MAX = 34, 48
SAVE_DIR = "osm_tiles_eMAS_z7"

HEADERS = {
    "User-Agent": "DSTPP-Research/1.0 (academic research)"
}

def lon_to_x(lon, zoom):
    return int((lon + 180) / 360 * (2 ** zoom))

def lat_to_y(lat, zoom):
    lat_rad = math.radians(lat)
    return int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * (2 ** zoom))

def download_tile(z, x, y, save_dir):
    path = Path(save_dir) / str(z) / str(x) / f"{y}.png"
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                f.write(r.content)
            return True
        else:
            print(f"  HTTP {r.status_code} for {z}/{x}/{y}")
            return False
    except Exception as e:
        print(f"  Error {z}/{x}/{y}: {e}")
        return False

x_min = lon_to_x(LON_MIN, ZOOM)
x_max = lon_to_x(LON_MAX, ZOOM)
y_min = lat_to_y(LAT_MAX, ZOOM)  # lat_max → smaller y
y_max = lat_to_y(LAT_MIN, ZOOM)  # lat_min → larger y

total = (x_max - x_min + 1) * (y_max - y_min + 1)
print(f"Zoom={ZOOM}: x=[{x_min},{x_max}], y=[{y_min},{y_max}]")
print(f"Total tiles to download: {total}")

count = 0
for x in range(x_min, x_max + 1):
    for y in range(y_min, y_max + 1):
        ok = download_tile(ZOOM, x, y, SAVE_DIR)
        count += 1
        print(f"  [{count}/{total}] tile {ZOOM}/{x}/{y} {'ok' if ok else 'FAILED'}")
        time.sleep(0.3)  # be polite to OSM

print(f"\nDone. Downloaded to: {SAVE_DIR}/")
