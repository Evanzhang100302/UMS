"""
Crawl OSM tiles for GOES-18 region at zoom 7.
GOES-18 fire+smoke: lon -121~-117, lat 32~35

Run: conda run -n stdpp python crawl_tiles_goes18.py
"""
import os, math, time, requests
from pathlib import Path

ZOOM    = 7
LON_MIN, LON_MAX = -122, -116
LAT_MIN, LAT_MAX = 31, 36
SAVE_DIR = "osm_tiles_GOES18_z7"

HEADERS = {"User-Agent": "DSTPP-Research/1.0 (academic research)"}

def lon_to_x(lon, zoom): return int((lon + 180) / 360 * (2 ** zoom))
def lat_to_y(lat, zoom):
    lat_rad = math.radians(lat)
    return int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * (2 ** zoom))

def download_tile(z, x, y):
    path = Path(SAVE_DIR) / str(z) / str(x) / f"{y}.png"
    if path.exists():
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            path.write_bytes(r.content)
            return True
        print(f"  HTTP {r.status_code} for {z}/{x}/{y}")
        return False
    except Exception as e:
        print(f"  Error {z}/{x}/{y}: {e}")
        return False

x_min = lon_to_x(LON_MIN, ZOOM)
x_max = lon_to_x(LON_MAX, ZOOM)
y_min = lat_to_y(LAT_MAX, ZOOM)
y_max = lat_to_y(LAT_MIN, ZOOM)
total = (x_max - x_min + 1) * (y_max - y_min + 1)
print(f"Zoom={ZOOM}: x=[{x_min},{x_max}], y=[{y_min},{y_max}], total={total}")

count = 0
for x in range(x_min, x_max + 1):
    for y in range(y_min, y_max + 1):
        ok = download_tile(ZOOM, x, y)
        count += 1
        print(f"  [{count}/{total}] {ZOOM}/{x}/{y} {'ok' if ok else 'FAILED'}")
        time.sleep(0.5)

print(f"Done → {SAVE_DIR}/")
