"""
Dukan Reservoir, Lesser Zab River (Iraq): satellite record, level
reconstruction, and machine-learning forecasts of reservoir surface area.

Inputs (same folder):
  Dukan_landsat_area.csv, Dukan_s1_area.csv, Dukan_jrc_area.csv,
  Dukan_rain.csv, Dukan_climate.csv, Dukan_snow.csv   (Google Earth Engine exports)
  dahiti_dukan_levels.csv                              (DAHITI altimetry, target 38593)

Outputs:
  results.json                      every number reported in the paper
  results_metrics.csv               cross-validated skill table
  Dukan_area_cleaned.csv            merged monthly surface-area record
  Dukan_level_reconstruction.csv    monthly level estimates
  fig2 ... fig8 (PNG, 300 dpi)

  pip install -r requirements.txt
  python 02_ml_pipeline.py
"""
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
BLUE, ORANGE, GREEN, RED, GREY = "#1f5f99", "#d98c1f", "#3a9a5b", "#b8433f", "0.45"
R = {}


def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return float(1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2))


def read(name, cols):
    d = pd.read_csv(name)
    d["date"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
    return d[["date"] + cols]


def rf():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)


def xgb():
    return XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8, random_state=42)


# ---------------------------------------------------------------- 1. load and merge
df = (read("Dukan_landsat_area.csv", ["landsat_area_km2", "n_landsat"])
      .merge(read("Dukan_s1_area.csv", ["s1_area_km2", "n_s1"]), on="date", how="outer")
      .merge(read("Dukan_jrc_area.csv", ["jrc_area_km2"]), on="date", how="left")
      .merge(read("Dukan_rain.csv", ["rain_mm"]), on="date", how="outer")
      .merge(read("Dukan_climate.csv", ["temp_K", "pot_evap_m", "snowmelt_m"]), on="date", how="outer")
      .merge(read("Dukan_snow.csv", ["snow_frac"]), on="date", how="outer")
      .sort_values("date").reset_index(drop=True))
R["n_months"] = len(df)

# ---------------------------------------------------------------- 2. surface-area record
df.loc[(df.n_landsat == 0) | (df.landsat_area_km2 == 0), "landsat_area_km2"] = np.nan
df.loc[(df.n_s1 == 0) | (df.s1_area_km2 == 0), "s1_area_km2"] = np.nan

both = df.dropna(subset=["landsat_area_km2", "s1_area_km2"]).copy()
both["err"] = (both.landsat_area_km2 - both.s1_area_km2).abs()
R["mae_le4_scenes"] = round(float(both[both.n_landsat <= 4].err.mean()), 1)
R["n_le4_scenes"] = int((both.n_landsat <= 4).sum())
R["mae_gt4_scenes"] = round(float(both[both.n_landsat > 4].err.mean()), 1)
R["n_gt4_scenes"] = int((both.n_landsat > 4).sum())

df["area"] = np.where(df.s1_area_km2.notna(), df.s1_area_km2, df.landsat_area_km2)
median7 = df["area"].rolling(7, center=True, min_periods=3).median()
low = (median7 - df["area"] > 20) & df.s1_area_km2.isna()
R["low_outliers_removed"] = int(low.sum())
df.loc[low, "area"] = np.nan
df["area"] = df["area"].interpolate(limit=2)
R["valid_months"] = int(df.area.notna().sum())
R["area_min"], R["area_max"] = round(float(df.area.min())), round(float(df.area.max()))
df["temp_C"] = df.temp_K - 273.15
df[["date", "area"]].to_csv("Dukan_area_cleaned.csv", index=False)

R["r_landsat_s1"], R["n_landsat_s1"] = round(float(both.landsat_area_km2.corr(both.s1_area_km2)), 2), len(both)
jrc = df.dropna(subset=["landsat_area_km2", "jrc_area_km2"])
R["r_landsat_jrc_all"] = round(float(jrc.landsat_area_km2.corr(jrc.jrc_area_km2)), 2)
jrc = jrc[jrc.jrc_area_km2 >= 50]
R["r_landsat_jrc"], R["n_landsat_jrc"] = round(float(jrc.landsat_area_km2.corr(jrc.jrc_area_km2)), 2), len(jrc)
R["n_jrc_incomplete"] = int((df.jrc_area_km2 < 50).sum())

annual = df.groupby(df.date.dt.year).agg(area=("area", "mean"), rain=("rain_mm", "sum"))
R["annual_mean_area"] = {int(y): round(float(v), 0) for y, v in annual.area.items()}
R["annual_rain_mm"] = {int(y): round(float(v), 0) for y, v in annual.rain.items()}

# ---------------------------------------------------------------- 3. level from area (DAHITI)
dah = pd.read_csv("dahiti_dukan_levels.csv", sep=";", parse_dates=["datetime"])
dah["date"] = dah.datetime.dt.to_period("M").dt.to_timestamp()
dah_m = dah.groupby("date").wse.mean().reset_index()
pairs = df[["date", "area"]].merge(dah_m, on="date").dropna()
coef = np.polyfit(pairs.area, pairs.wse, 2)
fitted = np.polyval(coef, pairs.area)
R["level_n_pairs"] = len(pairs)
R["level_r"] = round(float(pairs.area.corr(pairs.wse)), 2)
R["level_rmse_m"] = round(float(np.sqrt(np.mean((pairs.wse - fitted) ** 2))), 1)
R["level_range_m"] = round(float(pairs.wse.max() - pairs.wse.min()), 1)
R["level_within_1rmse_pct"] = round(float((np.abs(pairs.wse - fitted) <= R["level_rmse_m"]).mean() * 100))
R["level_within_2rmse_pct"] = round(float((np.abs(pairs.wse - fitted) <= 2 * R["level_rmse_m"]).mean() * 100))
R["dahiti_min"], R["dahiti_max"] = round(float(dah.wse.min()), 1), round(float(dah.wse.max()), 1)
df["level"] = np.polyval(coef, df.area)
R["level_est_min"], R["level_est_max"] = round(float(df.level.min()), 1), round(float(df.level.max()), 1)
for y0, y1, key in [(2021, 2023, "2021_2023"), (2025, 2025, "2025"), (2008, 2009, "2008_2009")]:
    w = df[(df.date.dt.year >= y0) & (df.date.dt.year <= y1)]
    R[f"level_min_{key}"] = round(float(w.level.min()), 1)
    R[f"area_min_{key}"] = round(float(w.area.min()))
df[["date", "area", "level"]].to_csv("Dukan_level_reconstruction.csv", index=False)

# ---------------------------------------------------------------- 4. predictors
base = ["rain_mm", "temp_C", "pot_evap_m", "snowmelt_m", "snow_frac"]
for c in base:
    for lag in (1, 2, 3, 6):
        df[f"{c}_l{lag}"] = df[c].shift(lag)
df["rain3"] = df.rain_mm.rolling(3).sum()
df["rain6"] = df.rain_mm.rolling(6).sum()
df["snow3"] = df.snow_frac.rolling(3).mean()
df["msin"] = np.sin(2 * np.pi * df.date.dt.month / 12)
df["mcos"] = np.cos(2 * np.pi * df.date.dt.month / 12)
df["area_l1"], df["area_l2"] = df.area.shift(1), df.area.shift(2)
F_CLIM = [f"{c}_l{l}" for c in base for l in (1, 2, 3, 6)] + base + ["rain3", "rain6", "snow3", "msin", "mcos"]
F_STATE = ["area", "area_l1", "area_l2"]
F = F_CLIM + F_STATE


def family(f):
    if f.startswith("snowmelt"): return "Snowmelt"
    if f.startswith("snow"): return "Snow cover"
    if f.startswith("rain"): return "Precipitation"
    if f.startswith("pot_evap"): return "Evaporation"
    if f.startswith("temp"): return "Temperature"
    if f in ("msin", "mcos"): return "Seasonality"
    return "Reservoir state"


rows = []
# 4a. climate-only estimate of absolute area (tests whether climate alone explains storage)
d = df.dropna(subset=F_CLIM + ["area"]).reset_index(drop=True)
for name, mk in [("RF", rf), ("XGB", xgb)]:
    s = [nse(d.loc[te, "area"], mk().fit(d.loc[tr, F_CLIM], d.loc[tr, "area"]).predict(d.loc[te, F_CLIM]))
         for tr, te in TimeSeriesSplit(5).split(d)]
    rows.append(dict(setting="climate_only_absolute", lead=0, model=name,
                     NSE=round(float(np.mean(s)), 2), NSE_sd=round(float(np.std(s)), 2), n=len(d)))

# 4a-2. sensitivity: add multi-year water-supply memory to the climate-only model
mem = df.copy()
mem["net"] = mem.rain_mm + 1000 * mem.snowmelt_m + 1000 * mem.pot_evap_m
extra = []
for w in (12, 24, 36):
    mem[f"rain{w}"] = mem.rain_mm.rolling(w).sum()
    mem[f"net{w}"] = mem.net.rolling(w).sum()
    extra += [f"rain{w}", f"net{w}"]
d = mem.dropna(subset=F_CLIM + extra + ["area"]).reset_index(drop=True)
s = [nse(d.loc[te, "area"], rf().fit(d.loc[tr, F_CLIM + extra], d.loc[tr, "area"]).predict(d.loc[te, F_CLIM + extra]))
     for tr, te in TimeSeriesSplit(5).split(d)]
R["climate_only_with_12_36mo_memory_nse"] = round(float(np.mean(s)), 2)

# 4b. change-based forecasts against persistence on identical folds
hold = {}
for lead in (1, 2, 3):
    d = df.copy()
    d["dy"] = d.area.shift(-lead) - d.area
    d["obs"] = d.area.shift(-lead)
    d = d.dropna(subset=F + ["dy", "obs"]).reset_index(drop=True)
    for name, mk in [("RF", rf), ("XGB", xgb)]:
        s = []
        for tr, te in TimeSeriesSplit(5).split(d):
            m = mk().fit(d.loc[tr, F], d.loc[tr, "dy"])
            s.append(nse(d.loc[te, "obs"], d.loc[te, "area"] + m.predict(d.loc[te, F])))
        rows.append(dict(setting="forecast", lead=lead, model=name,
                         NSE=round(float(np.mean(s)), 2), NSE_sd=round(float(np.std(s)), 2), n=len(d)))
    s = [nse(d.loc[te, "obs"], d.loc[te, "area"]) for _, te in TimeSeriesSplit(5).split(d)]
    rows.append(dict(setting="forecast", lead=lead, model="Persistence",
                     NSE=round(float(np.mean(s)), 2), NSE_sd=round(float(np.std(s)), 2), n=len(d)))
    # chronological holdout: last 20 % never seen in training
    sp = int(len(d) * 0.8)
    m = rf().fit(d.loc[:sp - 1, F], d.loc[:sp - 1, "dy"])
    pred = d.loc[sp:, "area"].values + m.predict(d.loc[sp:, F])
    R[f"holdout_nse_t{lead}"] = round(nse(d.loc[sp:, "obs"], pred), 2)
    R[f"holdout_persistence_nse_t{lead}"] = round(nse(d.loc[sp:, "obs"], d.loc[sp:, "area"]), 2)
    hold[lead] = (d.date[sp:] + pd.DateOffset(months=lead), d.loc[sp:, "obs"].values, pred)
    R["holdout_start"] = str((d.date[sp] + pd.DateOffset(months=1)).date())
metrics = pd.DataFrame(rows)
metrics.to_csv("results_metrics.csv", index=False)
R["metrics"] = rows

# ---------------------------------------------------------------- 5. SHAP on the forecast models
shares = {}
for lead in (1, 3):
    d = df.copy()
    d["dy"] = d.area.shift(-lead) - d.area
    d = d.dropna(subset=F + ["dy"]).reset_index(drop=True)
    sv = shap.TreeExplainer(rf().fit(d[F], d["dy"])).shap_values(d[F])
    imp = pd.Series(np.abs(sv).mean(0), index=F)
    R[f"shap_top5_t{lead}"] = imp.sort_values(ascending=False).head(5).round(2).to_dict()
    fam = imp.groupby(imp.index.map(family)).sum()
    R[f"shap_share_all_t{lead}"] = (fam / fam.sum() * 100).round(1).to_dict()
    clim = fam.drop("Reservoir state")
    shares[lead] = (clim / clim.sum() * 100).round(1)
    R[f"shap_share_climate_t{lead}"] = shares[lead].to_dict()

# ---------------------------------------------------------------- 6. upstream-development test
d = df.copy()
d["dy"] = d.area.shift(-1) - d.area
d = d.dropna(subset=F + ["dy"]).reset_index(drop=True)
pre = d[d.date < "2017-06-01"].reset_index(drop=True)
pre_res = np.full(len(pre), np.nan)
for tr, te in TimeSeriesSplit(5).split(pre):
    pre_res[te] = pre.loc[te, "dy"] - rf().fit(pre.loc[tr, F], pre.loc[tr, "dy"]).predict(pre.loc[te, F])
ok = ~np.isnan(pre_res)
R["attr_pre_oos_nse"] = round(nse(pre.dy[ok], pre.dy[ok] - pre_res[ok]), 2)
m = rf().fit(pre[F], pre["dy"])
post = d[d.date >= "2017-06-01"].reset_index(drop=True)
post["res"] = post.dy - m.predict(post[F])
r0, r1, r2 = pre_res[ok], post[post.date < "2020-10-01"].res, post[post.date >= "2020-10-01"].res
R["attr_mean_res_pre"] = round(float(r0.mean()), 1)
R["attr_mean_res_2017_2020"] = round(float(r1.mean()), 1)
R["attr_mean_res_post2020"] = round(float(r2.mean()), 1)
R["attr_n"] = [int(len(r0)), int(len(r1)), int(len(r2))]
R["attr_p_pre_vs_post2020"] = round(float(stats.mannwhitneyu(r0, r2).pvalue), 2)
R["attr_p_2017_vs_post2020"] = round(float(stats.mannwhitneyu(r1, r2).pvalue), 2)
R["attr_p_post2020_mean_zero"] = round(float(stats.ttest_1samp(r2, 0).pvalue), 2)
R["attr_kruskal_p"] = round(float(stats.kruskal(r0, r1, r2).pvalue), 2)

json.dump(R, open("results.json", "w"), indent=2, default=str)

# ---------------------------------------------------------------- 7. figures
BREAKS = [(pd.Timestamp("2017-06-01"), "Sardasht impounding"), (pd.Timestamp("2020-10-01"), "Kani Sib transfer")]


def label(ax, i):
    ax.text(0.005, 0.94, f"({'abcdefgh'[i]})", transform=ax.transAxes, fontweight="bold", va="top")


fig, ax = plt.subplots(4, 1, figsize=(7.2, 7.2), sharex=True)
ax[0].plot(df.date, df.area, color=BLUE, lw=1)
ax[0].set_ylabel("Surface area\n(km²)")
for t, _ in BREAKS:
    ax[0].axvline(t, ls="--", color=GREY, lw=0.8)
ax[1].bar(df.date, df.rain_mm, width=25, color="#4a9fd4")
ax[1].set_ylabel("Precipitation\n(mm month⁻¹)")
ax[2].plot(df.date, df.snow_frac * 100, color="#6b5fa8", lw=0.9)
ax[2].set_ylabel("Snow cover\n>1500 m (%)")
ax[3].plot(df.date, df.temp_C, color=RED, lw=0.9)
ax[3].set_ylabel("Air temperature\n(°C)")
for i, a in enumerate(ax):
    label(a, i)
fig.align_ylabels(ax)
plt.savefig("fig2_record.png"); plt.close()

fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.plot(df.date, df.landsat_area_km2, lw=0.9, color=BLUE, label="Landsat (MNDWI)")
ax.plot(df.date, df.s1_area_km2, lw=0.9, color=ORANGE, label="Sentinel-1 (VV)")
ax.plot(df.date, df.jrc_area_km2.where(df.jrc_area_km2 >= 50), lw=0.9, color=GREEN, alpha=0.85,
        label="JRC monthly water history")
ax.set_ylabel("Water area (km²)")
ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=8)
plt.savefig("fig3_sensors.png"); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 2.2]})
ax[0].scatter(pairs.area, pairs.wse, s=11, color=BLUE)
xs = np.linspace(pairs.area.min(), pairs.area.max(), 100)
ax[0].plot(xs, np.polyval(coef, xs), color=RED)
ax[0].set_xlabel("Surface area (km²)")
ax[0].set_ylabel("Water level (m a.s.l.)")
ax[1].plot(df.date, df.level, color=BLUE, lw=0.9, label="Estimated from surface area")
ax[1].scatter(dah.datetime, dah.wse, s=8, color="k", zorder=3, label="Envisat altimetry (DAHITI)")
ax[1].set_ylabel("Water level (m a.s.l.)")
ax[1].legend(frameon=False, fontsize=8, loc="lower left")
for i, a in enumerate(ax):
    label(a, i)
plt.savefig("fig4_level.png"); plt.close()

fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
for i, lead in enumerate((1, 3)):
    t, obs, pred = hold[lead]
    ax[i].plot(df.date[df.date >= t.iloc[0] - pd.DateOffset(years=3)],
               df.area[df.date >= t.iloc[0] - pd.DateOffset(years=3)], color="k", lw=1, label="Observed")
    ax[i].plot(t, pred, color=RED, lw=1.3, ls="--",
               label=f"{lead}-month forecast (NSE = {R[f'holdout_nse_t{lead}']:.2f})")
    ax[i].axvline(t.iloc[0], color=GREY, ls=":")
    ax[i].set_ylabel("Surface area (km²)")
    ax[i].legend(frameon=False, fontsize=8, loc="lower left")
    label(ax[i], i)
plt.savefig("fig6_holdout.png"); plt.close()

fc = metrics[metrics.setting == "forecast"]
fig, ax = plt.subplots(figsize=(4.0, 2.9))
style = {"RF": ("o", BLUE, "-", "Random forest"), "XGB": ("s", ORANGE, "-", "XGBoost"),
         "Persistence": ("^", GREY, "--", "Persistence")}
for mdl, g in fc.groupby("model"):
    mk, col, ls, lab = style[mdl]
    ax.plot(g.lead, g.NSE, marker=mk, color=col, ls=ls, label=lab)
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks([1, 2, 3])
ax.set_xlabel("Lead time (months)")
ax.set_ylabel("NSE")
ax.legend(frameon=False, fontsize=8)
plt.savefig("fig5_skill.png"); plt.close()

order = ["Precipitation", "Snow cover", "Snowmelt", "Evaporation", "Temperature"]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
x = np.arange(len(order))
ax.bar(x - 0.2, [shares[1][k] for k in order], 0.4, color=BLUE, label="1-month forecast")
ax.bar(x + 0.2, [shares[3][k] for k in order], 0.4, color=ORANGE, label="3-month forecast")
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_ylabel("Share of climate influence (%)")
ax.legend(frameon=False, fontsize=8)
plt.savefig("fig7_drivers.png"); plt.close()

fig, ax = plt.subplots(figsize=(7.2, 2.7))
ax.plot(post.date, post.res.cumsum(), color=RED, lw=1.1)
ax.axhline(0, color="k", lw=0.5)
for t, lab in BREAKS:
    ax.axvline(t, ls="--", color=GREY, lw=0.8)
    ax.text(t + pd.Timedelta(days=35), ax.get_ylim()[0] + 0.9 * np.diff(ax.get_ylim())[0], lab, fontsize=7.5)
ax.set_ylabel("Cumulative residual of\nmonthly change (km²)")
plt.savefig("fig8_upstream_test.png"); plt.close()

print(json.dumps({k: v for k, v in R.items() if k not in ("metrics",)}, indent=1, default=str))
print(metrics.to_string(index=False))
