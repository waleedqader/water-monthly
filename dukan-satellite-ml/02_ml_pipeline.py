"""
Dukan Reservoir ML pipeline — satellite-based reconstruction & forecasting.
Companion code for Qader (2026), Lesser Zab / Dukan reservoir study.

Inputs (same folder): Dukan_landsat_area.csv, Dukan_s1_area.csv,
  Dukan_jrc_area.csv, Dukan_rain.csv, Dukan_climate.csv, Dukan_snow.csv
Optional: dahiti_dukan_levels.csv (datetime;wse;wse_u) for level reconstruction.

Outputs: results_metrics.csv, baseline_comparison.csv, Dukan_area_cleaned.csv,
  and figures fig1..fig7 (PNG).

Run:  pip install pandas numpy scikit-learn xgboost shap matplotlib scipy
      python 02_ml_pipeline.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

def nse(o, s):
    o, s = np.asarray(o), np.asarray(s)
    return 1 - np.sum((o-s)**2) / np.sum((o-np.mean(o))**2)

def rd(name, cols):
    d = pd.read_csv(name); d["date"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    return d[["date"] + [c for c in cols if c in d.columns]]

# ---- 1. LOAD & MERGE ----
ls   = rd("Dukan_landsat_area.csv", ["landsat_area_km2", "n_landsat"])
s1   = rd("Dukan_s1_area.csv",      ["s1_area_km2", "n_s1"])
jrc  = rd("Dukan_jrc_area.csv",     ["jrc_area_km2"])
rain = rd("Dukan_rain.csv",         ["rain_mm"])
clim = rd("Dukan_climate.csv",      ["temp_K", "pot_evap_m", "snowmelt_m"])
snow = rd("Dukan_snow.csv",         ["snow_frac"])

df = ls.merge(s1,"outer","date").merge(jrc,"left","date").merge(rain,"outer","date") \
       .merge(clim,"outer","date").merge(snow,"outer","date").sort_values("date").reset_index(drop=True)

# ---- 2. CLEAN TARGET (physics-informed: remove only downward partial-scene spikes) ----
ls_bad = (df["n_landsat"]==0) | (df["landsat_area_km2"]==0)
df.loc[ls_bad, "landsat_area_km2"] = np.nan
s1_bad = (df.get("n_s1",1)==0) | (df["s1_area_km2"]==0)
df.loc[s1_bad, "s1_area_km2"] = np.nan
df["area"] = np.where(df["s1_area_km2"].notna(), df["s1_area_km2"], df["landsat_area_km2"])
med = df["area"].rolling(7, center=True, min_periods=3).median()
df.loc[(med - df["area"] > 20) & (df["s1_area_km2"].isna()), "area"] = np.nan   # downward Landsat spikes only
df["area"] = df["area"].interpolate(limit=2)
df["temp_C"] = df["temp_K"] - 273.15
df[["date","area"]].to_csv("Dukan_area_cleaned.csv", index=False)
print(f"Cleaned series: {df['area'].notna().sum()}/{len(df)} months valid")

# ---- 3. CROSS-SENSOR VALIDATION ----
both = df.dropna(subset=["landsat_area_km2","s1_area_km2"])
print(f"Landsat vs S1  r = {both['landsat_area_km2'].corr(both['s1_area_km2']):.3f} (n={len(both)})")
bj = df.dropna(subset=["landsat_area_km2","jrc_area_km2"])
print(f"Landsat vs JRC r = {bj['landsat_area_km2'].corr(bj['jrc_area_km2']):.3f} (n={len(bj)})")

# ---- 4. FEATURES ----
base = ["rain_mm","temp_C","pot_evap_m","snowmelt_m","snow_frac"]
for c in base:
    for l in (1,2,3,6): df[f"{c}_l{l}"] = df[c].shift(l)
df["rain3"] = df["rain_mm"].rolling(3).sum(); df["rain6"] = df["rain_mm"].rolling(6).sum()
df["snow3"] = df["snow_frac"].rolling(3).mean()
df["msin"] = np.sin(2*np.pi*df["date"].dt.month/12); df["mcos"] = np.cos(2*np.pi*df["date"].dt.month/12)
df["ar1"] = df["area"].shift(1); df["ar2"] = df["area"].shift(2)
F = [f"{c}_l{l}" for c in base for l in (1,2,3,6)] + base + ["rain3","rain6","snow3","msin","mcos","ar1","ar2"]

def mk_rf(): return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
def mk_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)

# ---- 5. RECONSTRUCTION (t+0) + CHANGE-BASED FORECASTS vs PERSISTENCE ----
rows = []
# reconstruction
d = df.dropna(subset=F+["area"]).reset_index(drop=True)
for name, mk in [("RF",mk_rf),("XGB",mk_xgb)]:
    ns=[]
    for tr,te in TimeSeriesSplit(5).split(d):
        m=mk(); m.fit(d.loc[tr,F], d.loc[tr,"area"]); p=m.predict(d.loc[te,F])
        ns.append(nse(d.loc[te,"area"].values, p))
    rows.append({"setting":"reconstruction","lead":0,"model":name,"NSE":np.mean(ns),"NSE_sd":np.std(ns)})
# forecasts (predict change, add back to last obs)
for lead in (1,2,3):
    dd = df.copy(); dd["ychange"] = dd["area"].shift(-lead) - dd["area"]; dd["ytrue"] = dd["area"].shift(-lead)
    dd = dd.dropna(subset=F+["ychange","ytrue"]).reset_index(drop=True)
    for name, mk in [("RF",mk_rf),("XGB",mk_xgb)]:
        ns=[]
        for tr,te in TimeSeriesSplit(5).split(dd):
            m=mk(); m.fit(dd.loc[tr,F], dd.loc[tr,"ychange"])
            pred = dd.loc[te,"area"].values + m.predict(dd.loc[te,F])
            ns.append(nse(dd.loc[te,"ytrue"].values, pred))
        rows.append({"setting":"forecast","lead":lead,"model":name,"NSE":np.mean(ns),"NSE_sd":np.std(ns)})
    # persistence on identical folds
    nsp=[]
    for tr,te in TimeSeriesSplit(5).split(dd):
        nsp.append(nse(dd.loc[te,"ytrue"].values, dd.loc[te,"area"].values))
    rows.append({"setting":"persistence","lead":lead,"model":"persistence","NSE":np.mean(nsp),"NSE_sd":np.std(nsp)})
res = pd.DataFrame(rows); res.to_csv("results_metrics.csv", index=False)
print("\n=== CROSS-VALIDATED RESULTS ===\n", res.round(3).to_string(index=False))

print("\nDone. results_metrics.csv + Dukan_area_cleaned.csv written.")
