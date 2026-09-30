"""Analysis for the companion paper on the seasonal cycle, trends and decadal extrapolation of SST.

Inputs
    data/sst_<Station>.csv                      observed daily SST (2000-2024)
    forecasts/forecast_2025_2034_<Station>.csv  monthly LSTM continuation written by run_pipeline.py

    python m2_analysis.py --data-dir data --forecast-dir results/forecasts --out-dir results_m2

Trends are estimated from annual means (OLS, Theil-Sen, Mann-Kendall). A straight line fitted to raw
monthly values that start in January and end in December is biased by the asymmetric seasonal cycle;
that slope and the slope of a repeated historical cycle are reported only to document the bias.
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
from scipy import stats

STATIONS = [
    ("SaintMartins", "Saint Martin's Island"), ("Teknaf", "Teknaf"), ("CoxsBazar", "Cox's Bazar"),
    ("Kutubdia", "Kutubdia"), ("Kuakata", "Kuakata"), ("DublarChar", "Dublar Char"), ("NijhumDwip", "Nijhum Dwip"),
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
LETTERS = "abcdefghi"
BLUE, ORANGE, RED = "#1f77b4", "#ff7f0e", "#d62728"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 10})


def load_observed(data_dir, code):
    df = pd.read_csv(os.path.join(data_dir, f"sst_{code}.csv"), parse_dates=["time"])
    return df.set_index("time")["SST_C"].resample("MS").mean().loc["2000-01-01":"2024-12-01"]


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


def main(data_dir, forecast_dir, out_dir, dpi):
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
    obs_rows, ext_rows, f25_rows, obs_fit, ext_fit, obs_ann, ext_ann = [], [], [], {}, {}, {}, {}
    for n in names:
        a, t, fit = annual_trend(obs[n]); obs_ann[n], obs_fit[n] = a, fit
        obs_rows.append(dict(Station=n, **t, Mean_2000_2004=a.loc[2000:2004].mean(), Mean_2020_2024=a.loc[2020:2024].mean(),
                             Record_max=obs[n].max(), Record_month=obs[n].idxmax().strftime("%Y-%m")))
        a2, t2e, fit2 = annual_trend(ext[n]); ext_ann[n], ext_fit[n] = a2, fit2
        ext_rows.append(dict(Station=n, Mean_2025=a2.loc[2025], Mean_2034=a2.loc[2034], Change=a2.loc[2034] - a2.loc[2025],
                             **t2e, Raw_monthly_slope=raw_slope(ext[n].values),
                             Repeated_cycle_slope=raw_slope(np.tile(cycle[n].values, 10))))
        f25 = ext[n].loc["2025"]; dep = f25.values - cycle[n].values
        f25_rows.append(dict(Station=n, Trough=f25.min(), Trough_month=MONTHS[f25.values.argmin()], Peak=f25.max(),
                             Peak_month=MONTHS[f25.values.argmax()], Mean_2025=f25.mean(), Departure=f25.mean() - obs[n].mean(),
                             Max_monthly_departure=dep.max(), Max_dep_month=MONTHS[dep.argmax()],
                             Min_monthly_departure=dep.min(), Min_dep_month=MONTHS[dep.argmin()]))
    reg_obs = pd.concat(obs, axis=1).mean(axis=1); reg_ext = pd.concat(ext, axis=1).mean(axis=1)
    a, t, fit = annual_trend(reg_obs)
    obs_rows.append(dict(Station="Regional mean", **t, Mean_2000_2004=a.loc[2000:2004].mean(), Mean_2020_2024=a.loc[2020:2024].mean()))
    a2, t2e, fit2 = annual_trend(reg_ext)
    ext_rows.append(dict(Station="Regional mean", Mean_2025=a2.loc[2025], Mean_2034=a2.loc[2034], Change=a2.loc[2034] - a2.loc[2025],
                         **t2e, Raw_monthly_slope=raw_slope(reg_ext.values),
                         Repeated_cycle_slope=raw_slope(np.tile(cycle.mean(axis=1).values, 10))))
    t2.round(3).to_csv(os.path.join(tab_dir, "table2_seasonal_cycle.csv"), index=False)
    pd.DataFrame(obs_rows).round(4).to_csv(os.path.join(tab_dir, "table3_observed_trends.csv"), index=False)
    pd.DataFrame(f25_rows).round(3).to_csv(os.path.join(tab_dir, "table4_forecast_2025.csv"), index=False)
    pd.DataFrame(ext_rows).round(4).to_csv(os.path.join(tab_dir, "table5_extrapolation.csv"), index=False)
    sd.round(3).to_csv(os.path.join(tab_dir, "monthly_interannual_sd.csv"))
    cycle.round(3).to_csv(os.path.join(tab_dir, "mean_seasonal_cycle.csv"))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(10, 6))                                   # Fig. 2
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for n, mk in zip(names, markers):
        ax.plot(MONTHS, cycle[n], marker=mk, label=n)
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
    print("tables and figures written to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--forecast-dir", default="results/forecasts")
    ap.add_argument("--out-dir", default="results_m2")
    ap.add_argument("--dpi", type=int, default=300)
    args, _ = ap.parse_known_args()
    main(args.data_dir, args.forecast_dir, args.out_dir, args.dpi)
