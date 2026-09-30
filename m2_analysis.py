"""Analysis for the companion paper on the seasonal cycle, trends and decadal extrapolation of SST.

Inputs
    data/sst_<Station>.csv                      observed daily SST (2000-2024)
    forecasts/forecast_2025_2034_<Station>.csv  monthly LSTM continuation written by run_pipeline.py

    python m2_analysis.py --data-dir data --forecast-dir results/forecasts --out-dir results_m2
    python m2_analysis.py ... --verify-dir data_2025        # adds the 2025 verification

Trends of the observed annual means are estimated by OLS with Newey-West (HAC) standard errors and by the
Theil-Sen slope with the Hamed-Rao modified Mann-Kendall test; the original, trend-free pre-whitened and
Yue-Wang Mann-Kendall variants are written as a sensitivity check. The 2025-2034 extrapolation is
deterministic model output, so its trends are reported without significance tests. A straight line
fitted to raw monthly values is biased by the seasonal phase structure; that slope and the slopes of
repeated seasonal cycles are reported to quantify the bias. Optional (not used in the paper): if
--verify-dir points to station files for 2025 in the same format as data/, the 2025 forecast is
verified against those observations; the 2000-2024 input files are not changed.
Figures carry panel labels but no titles, as required by the journal.
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

STATIONS = [
    ("SaintMartins", "Saint Martin's Island"), ("Teknaf", "Teknaf"), ("CoxsBazar", "Cox's Bazar"),
    ("Kutubdia", "Kutubdia"), ("Kuakata", "Kuakata"), ("DublarChar", "Dublar Char"), ("NijhumDwip", "Nijhum Dwip"),
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LETTERS = "abcdefghi"
BLUE, ORANGE, RED = "#1f77b4", "#ff7f0e", "#d62728"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 10})


def load_observed(data_dir, code, end="2024-12-01"):
    df = pd.read_csv(os.path.join(data_dir, f"sst_{code}.csv"), parse_dates=["time"])
    return df.set_index("time")["SST_C"].resample("MS").mean().loc["2000-01-01":end]


def load_forecast(forecast_dir, code):
    df = pd.read_csv(os.path.join(forecast_dir, f"forecast_2025_2034_{code}.csv"))
    return pd.Series(df["Forecast_SST_C"].values, index=pd.to_datetime(df["Date"].astype(str)))


def annual_trend(monthly):
    """Trend of annual means: OLS slope per decade with p-value, Theil-Sen slope, Mann-Kendall test."""
    ann = monthly.groupby(monthly.index.year).mean()
    yrs = ann.index.values.astype(float)
    ols = stats.linregress(yrs, ann.values)
    tau, p_mk = stats.kendalltau(yrs, ann.values)
    return ann, dict(OLS_per_decade=ols.slope * 10, OLS_p=ols.pvalue,
                     Sen_per_decade=stats.theilslopes(ann.values, yrs)[0] * 10, MK_tau=tau, MK_p=p_mk), ols


def observed_trend(monthly, hac_lags=2):
    """Annual-mean trend with autocorrelation-aware tests (observed record)."""
    ann = monthly.groupby(monthly.index.year).mean()
    y, yrs = ann.values, ann.index.values.astype(float)
    X = sm.add_constant(yrs)
    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    resid = y - ols.fittedvalues
    return ann, dict(
        OLS_per_decade=ols.params[1] * 10, OLS_p=ols.pvalues[1], OLS_p_NeweyWest=hac.pvalues[1],
        Sen_per_decade=stats.theilslopes(y, yrs)[0] * 10, MK_p=mk.original_test(y).p,
        MK_p_HamedRao=mk.hamed_rao_modification_test(y).p, MK_p_TFPW=mk.trend_free_pre_whitening_modification_test(y).p,
        MK_p_YueWang=mk.yue_wang_modification_test(y).p, Lag1_r=pd.Series(y).autocorr(1),
        Resid_lag1_r=pd.Series(resid).autocorr(1), Durbin_Watson=durbin_watson(resid)), stats.linregress(yrs, y)


def raw_slope(values):
    """Slope of a straight line through raw monthly values, per decade (biased by the seasonal cycle)."""
    return np.polyfit(np.arange(len(values)), values, 1)[0] * 120


def save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def legend_below(fig, ax, ncol=None):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol or len(l), frameon=False)


def station_grid(names, draw, path, dpi, ncol=None):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11.5))
    axes = list(axes.flat)
    for k, (ax, name) in enumerate(zip(axes, names)):
        draw(ax, name)
        ax.set_title(f"({LETTERS[k]}) {name}")
    for ax in axes[len(names):]:
        ax.axis("off")
    legend_below(fig, axes[0], ncol)
    save(fig, path, dpi)


def main(data_dir, forecast_dir, out_dir, dpi, verify_dir=None):
    fig_dir, tab_dir = os.path.join(out_dir, "figures"), os.path.join(out_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    names = [n for _, n in STATIONS]
    obs = {n: load_observed(data_dir, c) for c, n in STATIONS}
    ext = {n: load_forecast(forecast_dir, c) for c, n in STATIONS}
    cycle = pd.DataFrame({n: obs[n].groupby(obs[n].index.month).mean().values for n in names}, index=MONTHS)
    sd = pd.DataFrame({n: obs[n].groupby(obs[n].index.month).std().values for n in names}, index=MONTHS).T

    # ---------------- tables ----------------
    t2 = pd.DataFrame([dict(Station=n, Mean=obs[n].mean(), SD=obs[n].std(), Amplitude=cycle[n].max() - cycle[n].min(),
                            Min_month=cycle[n].idxmin(), Min=cycle[n].min(), Max_month=cycle[n].idxmax(), Max=cycle[n].max(),
                            May=cycle[n]["May"], Oct=cycle[n]["Oct"]) for n in names])
    obs_rows, stab_rows, cmp_rows, f25_rows, obs_fit, ext_fit, obs_ann, ext_ann = [], [], [], [], {}, {}, {}, {}
    amp = lambda s: s.groupby(s.index.year).apply(lambda v: v.max() - v.min())
    for n in names + ["Regional mean"]:
        o = obs[n] if n in obs else pd.concat(obs, axis=1).mean(axis=1)        # unweighted mean of the 7 stations
        e = ext[n] if n in ext else pd.concat(ext, axis=1).mean(axis=1)
        a, t, fit = observed_trend(o); obs_ann[n], obs_fit[n] = a, fit
        obs_rows.append(dict(Station=n, **t, Mean_2000_2004=a.loc[2000:2004].mean(), Mean_2020_2024=a.loc[2020:2024].mean(),
                             Change_2000_04_to_2020_24=a.loc[2020:2024].mean() - a.loc[2000:2004].mean(),
                             Record_max=o.max() if n in obs else np.nan, Record_month=o.idxmax().strftime("%Y-%m") if n in obs else ""))
        a2, t2e, fit2 = annual_trend(e); ext_ann[n], ext_fit[n] = a2, fit2
        ea = amp(e)
        stab_rows.append(dict(Station=n, Mean_2025=a2.loc[2025], Mean_2034=a2.loc[2034], Change=a2.loc[2034] - a2.loc[2025],
                              Range_of_annual_means=a2.max() - a2.min(), Amplitude_2025=ea.loc[2025], Amplitude_2034=ea.loc[2034],
                              Amplitude_change=ea.loc[2034] - ea.loc[2025], Mean_minus_2020_2024=e.mean() - o.loc["2020":"2024"].mean()))
        ocyc = o.groupby(o.index.month).mean().values; ecyc = e.groupby(e.index.month).mean().values
        anom = e - e.groupby(e.index.month).transform("mean")
        cmp_rows.append(dict(Station=n, Observed_annual_mean_trend=t["OLS_per_decade"], Extrapolation_annual_mean_trend=t2e["OLS_per_decade"],
                             Extrapolation_anomaly_trend=np.polyfit(np.arange(len(anom)), anom.values, 1)[0] * 120,
                             Raw_monthly_slope=raw_slope(e.values), Repeated_observed_cycle_slope=raw_slope(np.tile(ocyc, 10)),
                             Repeated_extrapolated_cycle_slope=raw_slope(np.tile(ecyc, 10))))
        if n in obs:
            f25 = e.loc["2025"]; dep = f25.values - ocyc
            f25_rows.append(dict(Station=n, Trough=f25.min(), Trough_month=MONTHS[f25.values.argmin()], Peak=f25.max(),
                                 Peak_month=MONTHS[f25.values.argmax()], Mean_2025=f25.mean(), Departure=f25.mean() - o.mean(),
                                 Max_monthly_departure=dep.max(), Max_dep_month=MONTHS[dep.argmax()],
                                 Min_monthly_departure=dep.min(), Min_dep_month=MONTHS[dep.argmin()]))
    a, fit, a2, fit2 = obs_ann["Regional mean"], obs_fit["Regional mean"], ext_ann["Regional mean"], ext_fit["Regional mean"]
    full = lambda df, name: df.to_csv(os.path.join(tab_dir, name), index=False, float_format="%.6f")
    t2.to_csv(os.path.join(tab_dir, "table2_seasonal_cycle.csv"), index=False, float_format="%.6f")
    full(pd.DataFrame(obs_rows), "table3_observed_trends.csv")
    full(pd.DataFrame(f25_rows), "table4_forecast_2025.csv")
    full(pd.DataFrame(stab_rows), "table5_extrapolation_stability.csv")
    full(pd.DataFrame(cmp_rows), "table6_trend_comparison.csv")

    # verification of the 2025 forecast, when observed data for 2025 are available
    ver_rows, ver = [], {}
    for c, n in (STATIONS if verify_dir else []):
        full_obs = load_observed(verify_dir, c, end="2025-12-01")
        o25 = full_obs[(full_obs.index >= "2025-01-01") & (full_obs.index <= "2025-12-01")]
        if len(o25) == 12 and o25.notna().all():
            f = ext[n].loc["2025"].values; y = o25.values; ver[n] = (o25, f)
            ver_rows.append(dict(Station=n, RMSE=np.sqrt(np.mean((f - y) ** 2)), MAE=np.mean(np.abs(f - y)),
                                 R2=1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2), Bias=np.mean(f - y)))
    if ver_rows:
        full(pd.DataFrame(ver_rows), "table_verification_2025.csv")
    sd.to_csv(os.path.join(tab_dir, "monthly_interannual_sd.csv"), float_format="%.6f")
    cycle.to_csv(os.path.join(tab_dir, "mean_seasonal_cycle.csv"), float_format="%.6f")

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(10, 6))                                   # Fig. 2
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for n, mrk in zip(names, markers):
        ax.plot(MONTHS, cycle[n], marker=mrk, label=n)
    ax.set_ylabel("SST (\u00b0C)"); ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, os.path.join(fig_dir, "Fig2_seasonal_cycle.png"), dpi)

    fig, ax = plt.subplots(figsize=(11, 4.6))                                 # Fig. 3
    im = ax.imshow(sd.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(MONTHS)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(sd.shape[0]):
        for j in range(12):
            ax.text(j, i, f"{sd.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, pad=0.02).set_label("Interannual SD (\u00b0C)")
    save(fig, os.path.join(fig_dir, "Fig3_monthly_variability.png"), dpi)

    def draw_annual(ax, n):                                                   # Fig. 4
        o, e = obs_ann[n], ext_ann[n]
        ax.plot(o.index, o.values, "o-", color=BLUE, ms=4, lw=1, label="Observed annual mean")
        ax.plot(o.index, obs_fit[n].intercept + obs_fit[n].slope * o.index.values, "--", color=BLUE, lw=1.4, label="Observed trend")
        ax.plot(e.index, e.values, "s-", color=ORANGE, ms=4, lw=1, label="Extrapolated annual mean")
        ax.plot(e.index, ext_fit[n].intercept + ext_fit[n].slope * e.index.values, "--", color=RED, lw=1.4, label="Extrapolation trend")
        ax.set_ylabel("Annual mean SST (\u00b0C)"); ax.grid(alpha=0.3)
    station_grid(names, draw_annual, os.path.join(fig_dir, "Fig4_annual_means.png"), dpi)

    def draw_2025(ax, n):                                                     # Fig. 5
        f25 = ext[n].loc["2025"]
        ax.plot(obs[n].index, obs[n].values, color=BLUE, lw=1.0, label="Observed 2000\u20132024")
        ax.plot(f25.index, f25.values, color=ORANGE, lw=1.8, label="Forecast 2025")
        ax.set_ylabel("SST (\u00b0C)"); ax.grid(alpha=0.3)
    station_grid(names, draw_2025, os.path.join(fig_dir, "Fig5_forecast_2025.png"), dpi)

    def draw_cycle(ax, n):                                                    # Fig. 6
        ax.plot(MONTHS, cycle[n], "o-", color=BLUE, label="Mean 2000\u20132024")
        ax.plot(MONTHS, ext[n].loc["2025"].values, "s-", color=ORANGE, label="Forecast 2025")
        ax.set_ylabel("SST (\u00b0C)"); ax.grid(alpha=0.3); ax.tick_params(axis="x", labelsize=9)
    station_grid(names, draw_cycle, os.path.join(fig_dir, "Fig6_cycle_vs_2025.png"), dpi)

    so = pd.concat(obs_ann, axis=1); se = pd.concat(ext_ann, axis=1)          # Fig. 7
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(so.index, so.min(axis=1), so.max(axis=1), color=BLUE, alpha=0.15, label="Station range (observed)")
    ax.fill_between(se.index, se.min(axis=1), se.max(axis=1), color=ORANGE, alpha=0.15, label="Station range (extrapolated)")
    ax.plot(a.index, a.values, "o-", color=BLUE, ms=4, lw=1, label="Observed regional mean")
    ax.plot(a.index, fit.intercept + fit.slope * a.index.values, "--", color=BLUE, lw=1.4, label="Observed trend")
    ax.plot(a2.index, a2.values, "s-", color=ORANGE, ms=4, lw=1, label="Extrapolated regional mean")
    ax.plot(a2.index, fit2.intercept + fit2.slope * a2.index.values, "--", color=RED, lw=1.4, label="Extrapolation trend")
    ax.set_ylabel("Annual mean SST (\u00b0C)"); ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    save(fig, os.path.join(fig_dir, "Fig7_regional_annual_means.png"), dpi)

    if ver:                                                                   # verification figure (2025)
        def draw_ver(ax, n):
            o25, f = ver[n]
            ax.plot(MONTHS, o25.values, "o-", color=BLUE, label="Observed 2025")
            ax.plot(MONTHS, f, "s--", color=ORANGE, label="LSTM forecast 2025")
            ax.set_ylabel("SST (\u00b0C)"); ax.grid(alpha=0.3); ax.tick_params(axis="x", labelsize=9)
        station_grid([n for n in names if n in ver], draw_ver, os.path.join(fig_dir, "Fig_verification_2025.png"), dpi)
    print("tables and figures written to", out_dir, "| 2025 verification:", "done" if ver else "skipped (no --verify-dir or incomplete 2025 data)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--forecast-dir", default="results/forecasts")
    ap.add_argument("--out-dir", default="results_m2")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--verify-dir", default=None, help="folder with station CSVs for 2025 (optional)")
    args, _ = ap.parse_known_args()
    main(args.data_dir, args.forecast_dir, args.out_dir, args.dpi, args.verify_dir)
