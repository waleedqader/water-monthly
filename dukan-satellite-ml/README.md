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
| `02_ml_pipeline.py` | Python pipeline: QC/cleaning, feature engineering, RF & XGBoost, 5-fold blocked time-series CV, change-based forecasts vs persistence |
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

- Reconstruction of monthly reservoir area from climate alone: NSE = 0.74 (5-fold blocked CV)
- Change-based forecasts beat persistence at every horizon (t+1 to t+3 months)
- Dominant driver: Zagros snowmelt lagged ~3 months (SHAP)
- Cross-sensor agreement: Landsat–Sentinel-1 r = 0.96; Landsat–JRC r = 0.93

## Data sources (all open)

Landsat C2 L2 (USGS), Sentinel-1 (ESA/Copernicus), JRC Global Surface Water,
CHIRPS, ERA5-Land, MODIS MOD10A1, SRTM, HydroSHEDS, DAHITI altimetry
(dahiti.dgfi.tum.de).

## License

Code: MIT. Derived datasets: CC-BY 4.0. Please cite the paper.

## Author

Waleed M. Qader — Salahaddin University–Erbil / GIS Lab Iraq
