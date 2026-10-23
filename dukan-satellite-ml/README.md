# Dukan Satellite ML

Satellite-based machine learning framework for reconstructing and forecasting
Dukan Reservoir dynamics (Lesser Zab River, Kurdistan Region of Iraq), 2002–2025.

Companion code and data for:

> Qader, W.M. (2026). *A Satellite-Based Machine Learning Framework for
> Reconstructing and Forecasting Reservoir Dynamics in a Data-Scarce
> Transboundary Basin: Dukan Reservoir, Lesser Zab River, Iraq (2002–2025).*
> Submitted to *Journal of Hydrology: Regional Studies*.

## Contents

| File | Description |
|---|---|
| `01_gee_extraction.js` | Google Earth Engine script: exports monthly reservoir area (Landsat MNDWI, Sentinel-1 SAR, JRC), CHIRPS rainfall, ERA5-Land climate, MODIS snow cover |
| `02_ml_pipeline.py` | Python pipeline: surface-area record, DAHITI area–level relation, RF & XGBoost forecasts vs persistence (expanding-window CV and 2021–2025 holdout), SHAP drivers, upstream-signal test, all figures |
| `20_gee_map_layers.js` | GEE export of the layers for the study-area map |
| `make_study_area_map.py` | Builds Figure 1 from the exported layers and Natural Earth |
| `dahiti_dukan_levels.csv` | Envisat water levels for Dukan from DAHITI (target 38593) |
| `Dukan_level_reconstruction.csv` | Monthly water level estimated from surface area, 2002–2025 |
| `results.json`, `results_metrics.csv` | Every number reported in the paper |
| `fig2_record.png` … `fig8_upstream_test.png` | Figures 2–8 |
| `19_gee_map_export.js` | GEE export of study-area layers (basin, rivers, reservoir, snow zone, DEM) for cartography |
| `Dukan_landsat_area.csv` … `Dukan_snow.csv` | The six monthly GEE exports (288 months, 2002–2025) |
| `Dukan_area_cleaned.csv` | Merged, QC-cleaned monthly reservoir surface area |

## Reproduce the results

```bash
pip install -r requirements.txt
python 02_ml_pipeline.py
```

Prints the cross-validated metrics and writes `results_metrics.csv` and
`Dukan_area_cleaned.csv`. The GEE script runs at https://code.earthengine.google.com
with a free Earth Engine account.

## Key results

- Monthly surface area 2002–2025 ranged 73–220 km²; Landsat agrees with Sentinel-1 (r = 0.96) and JRC (r = 0.93)
- Area–level relation from Envisat altimetry: r = 0.83, RMSE 4.4 m; lowest stands in 2008–09 and late 2025
- Climate predictors alone cannot reproduce storage (NSE < 0)
- With current state, forecasts beat persistence at 1–3 months (NSE 0.85/0.65/0.52 vs 0.80/0.45/−0.05)
- No significant climate-normalised storage change after the Kani Sib transfer (p = 0.44)

## Data sources (all open)

Landsat C2 L2 (USGS), Sentinel-1 (ESA/Copernicus), JRC Global Surface Water,
CHIRPS, ERA5-Land, MODIS MOD10A1, SRTM, HydroSHEDS, DAHITI altimetry
(dahiti.dgfi.tum.de).

## License

Code: MIT. Derived datasets: CC-BY 4.0. Please cite the paper.

## Author

Waleed M. Qader — Salahaddin University–Erbil / GIS Lab Iraq
