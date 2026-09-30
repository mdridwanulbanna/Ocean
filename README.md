# Station-specific LSTM forecasting and seasonal analysis of SST along the Bangladesh coast

Code and data for two papers based on 25 years (2000–2024) of Level-4 satellite sea surface
temperature (SST) at seven coastal and island stations in the northern Bay of Bengal
(Saint Martin's Island, Teknaf, Cox's Bazar, Kutubdia, Kuakata, Dublar Char and Nijhum Dwip).

**Paper 1** (reproduced by `run_pipeline.py`)

> Rifat, M.R.B., Uddin, M.M., Farzin, N.A. *Station-specific RNN-LSTM models for forecasting sea surface
> temperature along the Bangladesh coast: development, validation, and cross-station performance.* (under review)

**Paper 2** (reproduced by `m2_analysis.py`, using the forecasts written by `run_pipeline.py`)

> Rifat, M.R.B., Uddin, M.M., Farzin, N.A., Hasnat, M.J. *Seasonal cycle, recent trends and LSTM-based
> extrapolation of coastal sea surface temperature at seven stations along the Bangladesh coast, northern
> Bay of Bengal.* (under review)

The study-area map used as Fig. 1 in both papers is produced by `make_study_area_map.py`.

## Repository layout

```
├── run_pipeline.py            Paper 1: LSTM models, baselines, tables and figures; 2025–2034 forecasts
├── m2_analysis.py             Paper 2: seasonal cycle, variability, trends, 2025 forecast, extrapolation
├── make_study_area_map.py     Fig. 1 of both papers
├── sst_forecast/
│   ├── config.py              stations, split dates and model settings (Paper 1, Table 2)
│   ├── data.py                loading and input windows
│   ├── models.py              LSTM, seeding, recursive forecast
│   ├── baselines.py           persistence, climatology, SARIMA
│   ├── metrics.py             accuracy metrics, climatology, correlations
│   └── plots.py               figures of Paper 1
├── data/                      station CSV files (see data/README.md)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

## Data

The input is one CSV file per station with the daily SST series. The source product, file format and
grid-cell coordinates are described in [data/README.md](data/README.md).

## Running

```bash
python run_pipeline.py --data-dir data --out-dir results                                    # Paper 1
python m2_analysis.py --data-dir data --forecast-dir results/forecasts --out-dir results_m2 # Paper 2
python make_study_area_map.py --data-dir mapdata --out Fig1_study_area.png --dpi 600        # Fig. 1
```

A quick check that everything works (not the published results):

```bash
python run_pipeline.py --data-dir data --out-dir test_run --epochs 2
```

**Google Colab:** upload the repository and the CSV files, select a GPU runtime, then run
`!pip install -r requirements.txt` followed by the commands above (each prefixed with `!`).
If a GPU run stops with a TensorFlow determinism error, add `--no-determinism` to `run_pipeline.py`.
A full run of `run_pipeline.py` takes roughly half an hour or more on a CPU and is considerably
faster on a GPU.

## Paper 1: what the pipeline does

1. Averages the daily station series to 300 monthly values per station.
2. Splits each series by date: training 2000–2021, validation 2022 (used only to monitor the training
   loss), test 2023–2024. Min-Max scaling is fitted on the training period only.
3. Trains one LSTM per station (60-month input window, 64 units, Adam, 100 epochs, batch size 12, seed 42).
   The settings are fixed; there is no tuning and no early stopping.
4. Forecasts every test month one step ahead from observed inputs, and does the same for persistence,
   climatology and SARIMA (SARIMA parameters are estimated on 2000–2021 and then held fixed).
5. Computes MSE, RMSE, MAE and R², and correlates station accuracy with historical SST variability.
6. Writes a recursive continuation of each LSTM from January 2025 to December 2034 (used in Paper 2).

| Paper 1 | File in `results/` |
|---|---|
| Table 3 (station climatology) | `tables/table3_climatology.csv` |
| Table 4 (LSTM test metrics) | `tables/table4_lstm_test_metrics.csv` |
| Table 5 (RMSE by station and method) | `tables/table5_rmse_by_station.csv` |
| Table 6 (mean accuracy by method) | `tables/table6_mean_accuracy.csv` |
| Table 7 (variability–accuracy correlations) | `tables/table7_correlations.csv` |
| Percentages in Section 3.4 | `tables/relative_rmse_differences.csv` |
| Fig. 4 (training and validation loss) | `figures/fig4_loss_curves.png` |
| Fig. 5 (test-period forecasts) | `figures/fig5_actual_vs_forecast.png` |

Also written: `figures/figS1_variability_vs_accuracy.png`, `figures/figS2_rmse_by_method.png`,
single-station versions of Figs. 4 and 5 in `per_station/`, the 2025–2034 forecasts in `forecasts/`,
a SARIMA convergence check in `tables/sarima_convergence.csv`, and the software versions of the run
in `run_info.txt`. The workflow and LSTM cell diagrams (Figs. 2–3) were drawn separately.

## Paper 2: what the analysis does

`m2_analysis.py` reads the observed series in `data/` and the forecasts written by `run_pipeline.py`. It
describes the mean seasonal cycle and the interannual variability of each calendar month, estimates the
2000–2024 trends from annual means (OLS, Theil–Sen, Mann–Kendall), and summarises the 2025 forecast and
the 2025–2034 extrapolation. A straight line fitted to raw monthly values is biased by the seasonal
cycle, so that slope is reported only to document the bias.

| Paper 2 | File in `results_m2/` |
|---|---|
| Table 2 (seasonal cycle) | `tables/table2_seasonal_cycle.csv` |
| Table 3 (observed trends, 2000–2024) | `tables/table3_observed_trends.csv` |
| Table 4 (forecast for 2025) | `tables/table4_forecast_2025.csv` |
| Table 5 (extrapolation, 2025–2034) | `tables/table5_extrapolation.csv` |
| Fig. 2 (mean seasonal cycle) | `figures/Fig2_seasonal_cycle.png` |
| Fig. 3 (interannual variability by month) | `figures/Fig3_monthly_variability.png` |
| Fig. 4 (annual means, observed and extrapolated) | `figures/Fig4_annual_means.png` |
| Fig. 5 (2025 forecast) | `figures/Fig5_forecast_2025.png` |
| Fig. 6 (seasonal cycle and 2025 forecast) | `figures/Fig6_cycle_vs_2025.png` |
| Fig. 7 (regional annual means) | `figures/Fig7_regional_annual_means.png` |

## Study-area map (Fig. 1 of both papers)

`make_study_area_map.py` draws the stations on Natural Earth 1:10m coastlines, rivers and borders
(public domain), limited to the study area (20.0–22.5°N, 89.0–92.5°E), with a scale bar computed for
the map latitude. The Natural Earth files are downloaded on first use.

## Reproducibility notes

- Random seeds are fixed (42) and TensorFlow op determinism is switched on, so repeated runs on the
  same machine give the same numbers. Results can still differ slightly between hardware and library
  versions; `run_info.txt` records the versions used.
- SARIMA estimates can change in the third decimal between statsmodels versions. The convergence of
  every SARIMA fit is written to `tables/sarima_convergence.csv`.
- The 2025–2034 series is a recursive statistical continuation of each model, not a climate projection.

## Citation

Please cite the relevant paper when it is published. Citation details for the code are in `CITATION.cff`.

## License

MIT (see `LICENSE`).

## Data acknowledgement

This study has been conducted using E.U. Copernicus Marine Service Information;
https://doi.org/10.48670/moi-00169.
