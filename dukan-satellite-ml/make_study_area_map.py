"""
Study-area map for the Dukan Reservoir paper (Figure 1).

Inputs (Earth Engine exports, script 20_gee_map_layers.js):
  map_dem.tif, map_hybas12.geojson, map_hybas7_predictors.geojson,
  map_rivers.geojson, map_lake.geojson
Natural Earth 1:10m (public domain): countries, land boundary lines, populated places.

Output: fig1_study_area.png (300 dpi) and basin_stats.json
"""
import json
import sys
from collections import defaultdict

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import Point, box

D = sys.argv[1] if len(sys.argv) > 1 else "."
NE = sys.argv[2] if len(sys.argv) > 2 else "."
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
EXT = (44.55, 46.55, 35.55, 37.05)          # lon_min, lon_max, lat_min, lat_max
DUKAN_DAM = (44.95278, 35.95417)            # Wikipedia, Dukan Dam infobox
SARDASHT_DAM = (45.566417, 36.081056)       # Wikipedia, Sardasht Dam infobox
EQ_AREA = "EPSG:32638"                      # UTM 38N for area statistics

# ------------------------------------------------------------ basin from HydroBASINS level 12
hb = gpd.read_file(f"{D}/map_hybas12.geojson")
hb["HYBAS_ID"] = hb.HYBAS_ID.astype("int64")
hb["NEXT_DOWN"] = hb.NEXT_DOWN.astype("int64")
outlet = hb[hb.contains(Point(DUKAN_DAM))]
if outlet.empty:                                   # dam on a unit edge: take nearest unit
    outlet = hb.iloc[[hb.distance(Point(DUKAN_DAM)).idxmin()]]
upstream = defaultdict(list)
for hid, nd in zip(hb.HYBAS_ID, hb.NEXT_DOWN):
    upstream[nd].append(hid)
keep, stack = set(), [int(outlet.HYBAS_ID.iloc[0])]
while stack:
    h = stack.pop()
    if h not in keep:
        keep.add(h)
        stack.extend(upstream.get(h, []))
basin_units = hb[hb.HYBAS_ID.isin(keep)]
basin = gpd.GeoDataFrame(geometry=[basin_units.unary_union], crs=hb.crs)

pred = gpd.read_file(f"{D}/map_hybas7_predictors.geojson")
pred_dom = gpd.GeoDataFrame(geometry=[pred.unary_union], crs=pred.crs)
b_eq, p_eq = basin.to_crs(EQ_AREA).geometry.iloc[0], pred_dom.to_crs(EQ_AREA).geometry.iloc[0]
stats = {"basin_units_level12": len(basin_units),
         "basin_area_km2": round(b_eq.area / 1e6),
         "predictor_domain_area_km2": round(p_eq.area / 1e6),
         "basin_share_inside_predictor_domain_pct": round(b_eq.intersection(p_eq).area / b_eq.area * 100, 1),
         "predictor_domain_share_inside_basin_pct": round(b_eq.intersection(p_eq).area / p_eq.area * 100, 1)}

# ------------------------------------------------------------ relief and snow zone
with rasterio.open(f"{D}/map_dem.tif") as src:
    dem = src.read(1).astype(float)
    dem[dem == src.nodata] = np.nan if src.nodata is not None else dem[dem == src.nodata]
    l, b, r, t = src.bounds
    tr = src.transform
dy_m = abs(tr.e) * 111_320
dx_m = abs(tr.a) * 111_320 * np.cos(np.radians(36.2))
ls = LightSource(azdeg=315, altdeg=40)
hs = ls.hillshade(np.nan_to_num(dem, nan=np.nanmin(dem)), vert_exag=1.5, dx=dx_m, dy=dy_m)
terrain = LinearSegmentedColormap.from_list(
    "t", ["#e7eedc", "#d9e3c4", "#e8dcb6", "#d8c19b", "#c4a785", "#b9a294", "#efece8"])

from rasterio.features import geometry_mask
inside = ~geometry_mask([basin.geometry.iloc[0]], dem.shape, tr)
snow = np.where(inside & (dem >= 1500), 1.0, np.nan)
stats["basin_share_above_1500m_pct"] = round(float(np.nansum(snow) / inside.sum() * 100), 1)
stats["basin_elev_min_m"] = int(np.nanmin(np.where(inside, dem, np.nan)))
stats["basin_elev_max_m"] = int(np.nanmax(np.where(inside, dem, np.nan)))
stats["basin_elev_mean_m"] = int(np.nanmean(np.where(inside, dem, np.nan)))

# ------------------------------------------------------------ vectors
clip = box(EXT[0], EXT[2], EXT[1], EXT[3])
rivers = gpd.read_file(f"{D}/map_rivers.geojson").clip(clip)
lake = gpd.read_file(f"{D}/map_lake.geojson")
lake["a"] = lake.to_crs(EQ_AREA).area
lake = lake[lake.a > 2e5]                     # drop specks under 0.2 km2
countries = gpd.read_file(f"{NE}/countries.geojson")
border = gpd.read_file(f"{NE}/boundary_lines.geojson").clip(box(40, 30, 50, 40))
places = gpd.read_file(f"{NE}/places.geojson")
places = places[places.intersects(clip)]

# ------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(7.2, 6.0))
ax.imshow(dem, extent=(l, r, b, t), cmap=terrain, vmin=250, vmax=3200, zorder=0)
ax.imshow(hs, extent=(l, r, b, t), cmap="gray", alpha=0.38, zorder=1)
ax.contourf(np.linspace(l, r, dem.shape[1]), np.linspace(t, b, dem.shape[0]), snow,
            levels=[0.5, 1.5], colors="none", hatches=["////"], zorder=2)
pred_dom.boundary.plot(ax=ax, color="#7a3e9d", lw=0.9, ls=(0, (4, 2)), zorder=3)
basin.boundary.plot(ax=ax, color="k", lw=1.4, zorder=4)
lw = 0.25 + 1.6 * (np.log10(rivers.UPLAND_SKM.clip(lower=100)) - 2) / 2
rivers.plot(ax=ax, color="#2b6cb0", linewidth=lw, zorder=5)
lake.plot(ax=ax, color="#2b6cb0", edgecolor="#1a4f8a", lw=0.3, zorder=6)
border.plot(ax=ax, color="#555", lw=1.0, ls=(0, (6, 2, 1, 2)), zorder=7)

ax.plot(*DUKAN_DAM, marker="^", ms=8, mfc="#b8433f", mec="k", mew=0.6, zorder=9)
ax.annotate("Dukan Dam", DUKAN_DAM, xytext=(-8, -12), textcoords="offset points", ha="right",
            fontsize=8, fontweight="bold", zorder=9)
ax.plot(*SARDASHT_DAM, marker="^", ms=7, mfc="#d98c1f", mec="k", mew=0.6, zorder=9)
ax.annotate("Sardasht Dam", SARDASHT_DAM, xytext=(6, -10), textcoords="offset points", fontsize=8, zorder=9)
for _, p in places.iterrows():
    x, y = p.geometry.x, p.geometry.y
    ax.plot(x, y, "o", ms=3.5, mfc="white", mec="k", mew=0.7, zorder=8)
    name = "Sulaymaniyah" if "Sulaymaniyah" in p["name"] else p["name"]
    ax.annotate(name, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7.5, zorder=8)
ax.text(45.05, 36.1, "Dukan\nReservoir", fontsize=7, style="italic", color="#1a4f8a", ha="left", zorder=9)
ax.text(46.25, 36.95, "IRAN", fontsize=10, color="#444", fontweight="bold", ha="center")
ax.text(44.8, 35.65, "IRAQ", fontsize=10, color="#444", fontweight="bold", ha="center")

ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
ax.set_aspect(1 / np.cos(np.radians(36.3)))
ax.set_xticks(np.arange(44.6, 46.6, 0.4)); ax.set_yticks(np.arange(35.6, 37.1, 0.4))
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}°E")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}°N")
ax.tick_params(labelsize=7, direction="in", top=True, right=True)

# scale bar (50 km) and north arrow
km_deg = 1 / (111.32 * np.cos(np.radians(35.7)))
x0, y0 = 45.85, 35.62
ax.add_patch(Rectangle((x0, y0), 25 * km_deg, 0.012, fc="k", ec="k", zorder=10))
ax.add_patch(Rectangle((x0 + 25 * km_deg, y0), 25 * km_deg, 0.012, fc="white", ec="k", zorder=10))
for d, s in [(0, "0"), (25, "25"), (50, "50 km")]:
    ax.text(x0 + d * km_deg, y0 + 0.025, s, ha="center", fontsize=7, zorder=10)
ax.annotate("N", xy=(46.45, 36.97), xytext=(46.45, 36.83), ha="center", fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2), zorder=10)

handles = [Line2D([], [], color="k", lw=1.4, label=f"Basin upstream of Dukan Dam ({stats['basin_area_km2']:,} km²)"),
           Line2D([], [], color="#7a3e9d", lw=0.9, ls=(0, (4, 2)), label="Domain of climate predictors"),
           Patch(fc="none", ec="k", hatch="////", lw=0, label="Basin terrain above 1500 m"),
           Line2D([], [], color="#2b6cb0", lw=1.2, label="Rivers (HydroRIVERS)"),
           Patch(fc="#2b6cb0", ec="#1a4f8a", label="Dukan Reservoir (JRC occurrence ≥ 10%)"),
           Line2D([], [], color="#555", lw=1.0, ls=(0, (6, 2, 1, 2)), label="International border"),
           Line2D([], [], marker="^", ls="", mfc="#b8433f", mec="k", label="Dam")]
leg = ax.legend(handles=handles, loc="lower left", fontsize=6.6, frameon=True, framealpha=0.9)
leg.get_frame().set_linewidth(0.4)

# elevation colour bar
sm = plt.cm.ScalarMappable(cmap=terrain, norm=plt.Normalize(250, 3200))
cax = fig.add_axes([0.915, 0.30, 0.015, 0.4])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("Elevation (m a.s.l.)", fontsize=7); cb.ax.tick_params(labelsize=6.5)

# location inset
ins = fig.add_axes([0.665, 0.655, 0.22, 0.22])
reg = countries.cx[37:50, 28:40]
reg.plot(ax=ins, color="#eeeeee", edgecolor="#999", lw=0.3)
countries[countries.ADMIN == "Iraq"].plot(ax=ins, color="#d9d2b4", edgecolor="#555", lw=0.5)
ins.add_patch(Rectangle((EXT[0], EXT[2]), EXT[1] - EXT[0], EXT[3] - EXT[2], fc="none", ec="#b8433f", lw=1.2))
ins.text(43.6, 33.0, "IRAQ", fontsize=6, ha="center"); ins.text(47.8, 33.6, "IRAN", fontsize=6, ha="center")
ins.set_xlim(38.5, 49.5); ins.set_ylim(29, 38.5); ins.set_xticks([]); ins.set_yticks([])
ins.set_facecolor("#dbe9f4")

fig.savefig("fig1_study_area.png")
json.dump(stats, open("basin_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
