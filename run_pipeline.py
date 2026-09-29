"""Reproduce the results of the paper.

    python run_pipeline.py --data-dir data --out-dir results

Outputs (in results/):
    tables/       Tables 3-7, relative RMSE differences, SARIMA convergence check
    figures/      Fig. 4 (loss curves), Fig. 5 (test forecasts), Fig. S1 and Fig. S2
    per_station/  the Fig. 4 and Fig. 5 panels as separate images
    forecasts/    monthly LSTM continuation from January 2025 to December 2034
    run_info.txt  software versions and settings of the run
"""
import argparse
import os
import platform
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
import numpy as np
import pandas as pd
import statsmodels
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from sst_forecast import baselines, plots
from sst_forecast.config import (BATCH_SIZE, EPOCHS, FORECAST_MONTHS, LEARNING_RATE, LOOK_BACK, SEED,
                                 STATIONS, TEST, TRAIN_END, UNITS, VAL)
from sst_forecast.data import load_monthly_sst, make_windows
from sst_forecast.metrics import climatology_summary, scores, variability_correlations
from sst_forecast.models import build_lstm, recursive_forecast, set_seed

METHODS = ["Persistence", "Climatology", "SARIMA", "LSTM"]


def run_station(code, name, args):
    monthly = load_monthly_sst(args.data_dir, code)

    # Min-Max scaling fitted on the training period only, then applied to every month
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(monthly.loc[:TRAIN_END].values.reshape(-1, 1))
    scaled = scaler.transform(monthly.values.reshape(-1, 1)).flatten()

    X, y, targets = make_windows(scaled, monthly.index, LOOK_BACK)
    train = targets <= pd.Timestamp(TRAIN_END)
    val = (targets >= pd.Timestamp(VAL[0])) & (targets <= pd.Timestamp(VAL[1]))
    test = (targets >= pd.Timestamp(TEST[0])) & (targets <= pd.Timestamp(TEST[1]))
    print(f"{name}: {train.sum()} training, {val.sum()} validation, {test.sum()} test windows", flush=True)

    # the 2022 validation loss is only monitored (no early stopping, no tuning)
    set_seed(SEED, deterministic=not args.no_determinism)
    model = build_lstm()
    history = model.fit(X[train], y[train], epochs=args.epochs, batch_size=BATCH_SIZE,
                        validation_data=(X[val], y[val]), shuffle=False, verbose=0)

    # one-step-ahead test forecasts: every input window contains observed values only
    lstm = scaler.inverse_transform(model.predict(X[test], verbose=0)).flatten()
    observed = monthly.loc[TEST[0]:TEST[1]].values
    sarima, converged = baselines.sarima_one_step(monthly)
    predictions = {
        "Persistence": baselines.persistence(monthly),
        "Climatology": baselines.climatology(monthly),
        "SARIMA": sarima,
        "LSTM": lstm,
    }

    future = recursive_forecast(model, scaled[-LOOK_BACK:], FORECAST_MONTHS)
    future = scaler.inverse_transform(future.reshape(-1, 1)).flatten()

    return dict(
        monthly=monthly,
        history=history.history,
        test_series=pd.Series(lstm, index=targets[test]),
        climatology=dict(Station=name, **climatology_summary(monthly)),
        lstm_scores=dict(Station=name, **scores(observed, lstm)),
        method_scores=[dict(Station=name, Method=m, **scores(observed, p)) for m, p in predictions.items()],
        converged=dict(Station=name, MLE_converged=converged),
        future=pd.Series(future, index=pd.date_range("2025-01-01", periods=FORECAST_MONTHS, freq="MS")),
    )


def write_tables(results, names, out):
    clim = pd.DataFrame([r["climatology"] for r in results.values()])
    lstm = pd.DataFrame([r["lstm_scores"] for r in results.values()]).sort_values("RMSE")
    methods = pd.DataFrame([row for r in results.values() for row in r["method_scores"]])
    clim.round(4).to_csv(os.path.join(out, "table3_climatology.csv"), index=False)
    lstm.round(4).to_csv(os.path.join(out, "table4_lstm_test_metrics.csv"), index=False)

    # Table 5 is printed to 3 decimals, and its Mean row is the mean of the printed values
    t5 = methods.pivot(index="Station", columns="Method", values="RMSE").loc[names, METHODS].round(3)
    t5.loc["Mean"] = t5.loc[names].mean().round(3)
    t5.to_csv(os.path.join(out, "table5_rmse_by_station.csv"))
    t6 = methods.groupby("Method")[["RMSE", "MAE", "R2"]].mean().loc[METHODS].round(3)
    t6["RMSE"] = t5.loc["Mean", METHODS].values
    t6.to_csv(os.path.join(out, "table6_mean_accuracy.csv"))

    m = t5.loc[names].mean()
    pd.DataFrame([
        ("LSTM lower than persistence (%)", (m.Persistence - m.LSTM) / m.Persistence * 100),
        ("LSTM lower than climatology (%)", (m.Climatology - m.LSTM) / m.Climatology * 100),
        ("SARIMA lower than persistence (%)", (m.Persistence - m.SARIMA) / m.Persistence * 100),
        ("SARIMA lower than climatology (%)", (m.Climatology - m.SARIMA) / m.Climatology * 100),
        ("SARIMA lower than LSTM (%)", (m.LSTM - m.SARIMA) / m.LSTM * 100),
        ("LSTM higher than SARIMA (%)", (m.LSTM - m.SARIMA) / m.SARIMA * 100),
        ("Stations where SARIMA RMSE < LSTM RMSE", int((t5.loc[names, "SARIMA"] < t5.loc[names, "LSTM"]).sum())),
        ("Stations where LSTM RMSE < climatology RMSE", int((t5.loc[names, "LSTM"] < t5.loc[names, "Climatology"]).sum())),
    ], columns=["Quantity", "Value"]).round(2).to_csv(os.path.join(out, "relative_rmse_differences.csv"), index=False)

    status = pd.DataFrame([r["converged"] for r in results.values()])
    status.to_csv(os.path.join(out, "sarima_convergence.csv"), index=False)
    if not status["MLE_converged"].all():
        print("WARNING: the SARIMA fit did not converge at:", list(status.loc[~status["MLE_converged"], "Station"]))

    corr, merged = variability_correlations(clim, lstm)
    corr.round(4).to_csv(os.path.join(out, "table7_correlations.csv"), index=False)
    return lstm, t5, corr, merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="folder with the sst_<Station>.csv files")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs (100 in the paper)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-determinism", action="store_true",
                    help="switch off TensorFlow op determinism (only if a GPU run raises a determinism error)")
    args, _ = ap.parse_known_args()   # also works inside Jupyter / Colab

    folders = {k: os.path.join(args.out_dir, k) for k in ["tables", "figures", "per_station", "forecasts"]}
    for path in folders.values():
        os.makedirs(path, exist_ok=True)

    names = [name for _, name in STATIONS]
    results = {name: run_station(code, name, args) for code, name in STATIONS}

    lstm, t5, corr, merged = write_tables(results, names, folders["tables"])

    for code, name in STATIONS:
        f = results[name]["future"]
        pd.DataFrame({"Date": f.index.strftime("%Y-%m"), "Forecast_SST_C": f.values}).to_csv(
            os.path.join(folders["forecasts"], f"forecast_2025_2034_{code}.csv"), index=False)

    histories = {n: r["history"] for n, r in results.items()}
    monthly_all = {n: r["monthly"] for n, r in results.items()}
    test_series = {n: r["test_series"] for n, r in results.items()}
    fig_dir, single = folders["figures"], folders["per_station"]
    plots.station_panels(names, plots.loss_panel(histories), os.path.join(fig_dir, "fig4_loss_curves.png"),
                         single, "loss_curve", args.dpi)
    plots.station_panels(names, plots.forecast_panel(monthly_all, test_series),
                         os.path.join(fig_dir, "fig5_actual_vs_forecast.png"), single, "actual_vs_forecast", args.dpi)
    plots.variability_scatter(merged, corr.to_dict("records"), os.path.join(fig_dir, "figS1_variability_vs_accuracy.png"), args.dpi)
    plots.rmse_bars(t5, names, METHODS, os.path.join(fig_dir, "figS2_rmse_by_method.png"), args.dpi)

    with open(os.path.join(args.out_dir, "run_info.txt"), "w") as fh:
        fh.write(f"Python {platform.python_version()} | TensorFlow {tf.__version__} | statsmodels {statsmodels.__version__} | "
                 f"NumPy {np.__version__} | pandas {pd.__version__} | matplotlib {matplotlib.__version__}\n")
        fh.write(f"Device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'} | seed {SEED} | "
                 f"op determinism {'off' if args.no_determinism else 'on'}\n")
        fh.write(f"lookback {LOOK_BACK}, units {UNITS}, learning rate {LEARNING_RATE}, epochs {args.epochs}, batch {BATCH_SIZE}\n")
        fh.write(f"training to {TRAIN_END}, validation {VAL[0]} to {VAL[1]}, test {TEST[0]} to {TEST[1]}\n")

    print("\nLSTM test metrics:\n", lstm.round(3).to_string(index=False))
    print("\nRMSE by station (Table 5):\n", t5.to_string())
    print("\nVariability-accuracy correlations (Table 7):\n", corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
