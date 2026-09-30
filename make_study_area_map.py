"""Study-area map (Fig. 1) for both papers.

Draws the seven stations on Natural Earth 1:10m coastlines and borders, limited to the study
area (20.0-22.5 N, 89.0-92.5 E), with a scale bar computed for the map latitude.

    python make_study_area_map.py --data-dir mapdata --out Fig1_study_area.png --dpi 600

The Natural Earth GeoJSON files are downloaded into --data-dir if they are not already there.
"""
import argparse
import json
import os
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path

NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{}.geojson"
LAYERS = ["ne_10m_land", "ne_10m_admin_0_boundary_lines_land", "ne_10m_rivers_lake_centerlines"]
EXTENT = (89.0, 92.5, 20.0, 22.5)          # lon_min, lon_max, lat_min, lat_max (study area)

# Site coordinates (Table 1) and label placement (dx, dy in degrees, horizontal alignment)
STATIONS = [
    ("Saint Martin\u2019s Island", 20.630, 92.320, (-0.07, -0.02, "right")),
    ("Teknaf", 20.820, 92.320, (-0.07, 0.00, "right")),
    ("Cox\u2019s Bazar", 21.430, 91.930, (-0.07, 0.00, "right")),
    ("Kutubdia", 21.830, 91.820, (-0.07, 0.03, "right")),
    ("Kuakata", 21.780, 90.150, (0.06, -0.10, "left")),
    ("Dublar Char", 21.740, 89.500, (0.00, -0.13, "center")),
    ("Nijhum Dwip", 22.120, 91.020, (0.07, 0.02, "left")),
]
LAND, SEA, BORDER, RIVER = "#e9e5dc", "#d4e6f1", "#555555", "#7fb2d6"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"]})


def load(data_dir, name):
    path = os.path.join(data_dir, name + ".geojson")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(NE_URL.format(name), path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["features"]


def intersects(coords, pad=0.5):
    arr = np.asarray(coords)
    return not (arr[:, 0].max() < EXTENT[0] - pad or arr[:, 0].min() > EXTENT[1] + pad or
                arr[:, 1].max() < EXTENT[2] - pad or arr[:, 1].min() > EXTENT[3] + pad)


def polygon_patch(rings, **kw):
    verts, codes = [], []
    for ring in rings:
        ring = np.asarray(ring)[:, :2]
        verts.extend(ring.tolist())
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(ring) - 2) + [Path.CLOSEPOLY])
    return PathPatch(Path(verts, codes), **kw)


def draw_lines(ax, features, **kw):
    for f in features:
        g = f["geometry"]
        if g is None:
            continue
        parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        for part in parts:
            if intersects(part):
                arr = np.asarray(part)
                ax.plot(arr[:, 0], arr[:, 1], **kw)


def km_per_degree_lon(lat):
    return 111.32 * np.cos(np.radians(lat))


def main(data_dir, out, dpi):
    lon0, lon1, lat0, lat1 = EXTENT
    fig, ax = plt.subplots(figsize=(6.85, 5.6))
    ax.set_facecolor(SEA)

    for f in load(data_dir, "ne_10m_land"):
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for rings in polys:
            if intersects(rings[0]):
                ax.add_patch(polygon_patch(rings, facecolor=LAND, edgecolor="#6b6b6b", linewidth=0.5, zorder=1))
    draw_lines(ax, load(data_dir, "ne_10m_rivers_lake_centerlines"), color=RIVER, linewidth=0.6, zorder=2)
    draw_lines(ax, load(data_dir, "ne_10m_admin_0_boundary_lines_land"), color=BORDER, linewidth=0.8,
               linestyle=(0, (4, 2)), zorder=3)

    halo = [pe.withStroke(linewidth=2.5, foreground="white")]
    for name, lat, lon, (dx, dy, ha) in STATIONS:
        ax.plot(lon, lat, "o", markersize=7, markerfacecolor="#c0392b", markeredgecolor="black",
                markeredgewidth=0.8, zorder=5)
        ax.text(lon + dx, lat + dy, name, fontsize=9, fontweight="bold", ha=ha, va="center",
                path_effects=halo, zorder=6)
    ax.text(90.55, 20.75, "Bay of Bengal", fontsize=11, style="italic", color="#2c5f86", ha="center")
    ax.text(90.05, 22.38, "BANGLADESH", fontsize=9, color="#444444", ha="center", path_effects=halo)
    ax.text(92.47, 21.15, "MYANMAR", fontsize=8, color="#444444", ha="right", path_effects=halo)

    # scale bar (km), computed for its own latitude
    sb_lat, sb_lon, length_km = 20.18, 89.15, 50
    dlon = length_km / km_per_degree_lon(sb_lat)
    for k, (a, b) in enumerate([(0, 0.5), (0.5, 1.0)]):
        ax.add_patch(Rectangle((sb_lon + a * dlon, sb_lat), (b - a) * dlon, 0.025,
                               facecolor="black" if k == 0 else "white", edgecolor="black", linewidth=0.8, zorder=6))
    for frac, lab in [(0, "0"), (0.5, "25"), (1.0, "50 km")]:
        ax.text(sb_lon + frac * dlon, sb_lat + 0.05, lab, fontsize=8, ha="center", va="bottom", zorder=6)

    # north arrow
    nx, ny = 89.25, 20.45
    ax.add_patch(Polygon([[nx, ny + 0.16], [nx - 0.045, ny], [nx, ny + 0.04], [nx + 0.045, ny]],
                         closed=True, facecolor="black", zorder=6))
    ax.text(nx, ny + 0.19, "N", fontsize=10, fontweight="bold", ha="center", va="bottom", zorder=6)

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1 / np.cos(np.radians((lat0 + lat1) / 2)))
    ax.set_xticks(np.arange(lon0, lon1 + 0.01, 0.5))
    ax.set_yticks(np.arange(lat0, lat1 + 0.01, 0.5))
    ax.set_xticklabels([f"{x:.1f}\u00b0E" for x in ax.get_xticks()], fontsize=8)
    ax.set_yticklabels([f"{y:.1f}\u00b0N" for y in ax.get_yticks()], fontsize=8)
    ax.grid(color="white", linewidth=0.5, alpha=0.8, zorder=0)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if out.lower().endswith(".png"):
        fig.savefig(out[:-4] + ".tif", dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("map written:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="mapdata")
    ap.add_argument("--out", default="Fig1_study_area.png")
    ap.add_argument("--dpi", type=int, default=600)
    args, _ = ap.parse_known_args()
    main(args.data_dir, args.out, args.dpi)
