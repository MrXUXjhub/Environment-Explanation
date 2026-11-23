# Temporal Modelling and Explainable Analysis of Greenspace Exposure

[中文说明](README.zh-CN.md)

Separate Python scripts implement OLS, Random Forest, CatBoost, LightGBM,
XGBoost and LSTM, followed by SHAP and partial dependence analysis. The revised
workflow excludes **trip volume and travel distance**, uses the same 13
predictors and target hours for all six models, and evaluates later-period
performance with training-only temporal validation.

The original single-file analysis is preserved in [legacy/](legacy/).

## Models and data

| Component | Specification |
| --- | --- |
| Models | OLS, RF, CatBoost, LightGBM, XGBoost, LSTM |
| Time predictors | Day of Week, Hour of Day |
| Air pollution | O3, PM2.5, PM10, SO2, NO2, CO |
| Weather | Temperature, Wind Speed, Pressure, Humidity, Precipitation |
| Outcomes, fitted separately | SGE, EGE, GiniStart, GiniEnd |
| Excluded predictors | ordercount, Trip Volume, Ln_Trip Volume, Travel Distance |

Only the 13-column whitelist in [config.py](config.py) enters model inputs.
Extra columns are not automatically selected or used as mobility substitutes.
SGE/EGE are labelled origin/destination GE in exported figures; the Gini
outcomes describe their corresponding spatial distributions.

Place the local hourly table at data/GE.csv. Required columns and timestamp
handling are described in [data/README.md](data/README.md). Raw observations,
model weights and generated results are not included in this code update and
are excluded from Git. The code does not recreate the joined hourly table.

The original project lists these upstream sources: [Mobike/Biendata](https://www.biendata.xyz/),
[ESA WorldCover](https://esa-worldcover.org/en), [CNEMC](https://www.cnemc.cn/)
and [CMDC](https://data.cma.cn/).

## Repository structure

~~~text
.
├── train_ols.py / train_rf.py / train_catboost.py
├── train_lightgbm.py / train_xgboost.py / train_lstm.py
├── shap_analysis.py          # All 13 importances and SHAP summary plot
├── shap_dependence.py        # SHAP dependence scatter plots
├── pdp_1d.py                 # One-dimensional partial dependence
├── pdp_2d.py                 # Nine pairs in a 3 × 3 layout
├── prepare_data.py           # Shared modelling table and temporal protocol
├── summarize_results.py      # Metrics recalculated from saved predictions
├── plot_predictions.py       # Shared axis labels and one legend
├── import_saved_results.py   # Import matching previous 13-predictor fits
├── run_all.py / self_check.py
├── config.py / runtime.py / data_protocol.py
├── training_common.py / explanation_common.py / plot_utils.py
├── data/README.md
├── tests/ / figure_qa/
├── requirements.txt / requirements_lstm.txt / requirements-test.txt
├── 使用说明.txt
└── legacy/                   # Unmodified original single-file code
~~~

Keep the directory together: each model has a separate entry point and shares
the protocol and utility modules.

## Installation

Use Python 3.12 and a separate virtual environment. From the repository root:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_lstm.txt
~~~

On Windows, activate with .venv\Scripts\activate instead. For table models and
explanations without LSTM, install requirements.txt. Pinned versions are
recorded in [environment_versions.json](environment_versions.json).

## Running the revised workflow

Check the code and temporal protocol without training real estimators:

~~~bash
python self_check.py
~~~

These checks use generated fixtures, so they run without the author CSV.
An additional original-data regression check runs only when the exact author
CSV is available. GitHub Actions runs these data-free checks using
requirements-test.txt; it does not train the six models.

Prepare the local table, train XGBoost and explain destination GE:

~~~bash
python prepare_data.py
python train_xgboost.py --targets EGE
python shap_analysis.py --targets EGE
python shap_dependence.py --targets EGE
python pdp_1d.py --targets EGE
python pdp_2d.py --targets EGE
~~~

Training scripts default to fitting all four outcomes separately. Run a
different train_*.py file to change the model. For the complete workflow:

~~~bash
python run_all.py --stage all
~~~

This command **trains models**, then summarizes metrics and generates
XGBoost explanations and the comparison figure. Individual stages are
prepare, check, fit, summary, explain and plots. The check stage needs the
local CSV or a prepared protocol, but does not fit real estimators.

Custom paths and partial runs:

~~~bash
python prepare_data.py --data /path/to/GE.csv --results my_results
python train_rf.py --results my_results --targets EGE
python shap_analysis.py --results my_results --targets EGE --model RF
~~~

Use the same --results directory for dependent commands. Each entry point
provides --help. For the full comparison, first fit all six models and four
outcomes, then run summarize_results.py and plot_predictions.py.

## Temporal validation

1. Sort unique hourly records. Reject duplicate hours and non-finite modelling
   values; do not impute across the complete dataset.
2. Reserve the final 20% of original records as the reference test period.
3. Require continuous current-plus-previous-23-hour inputs at each target time.
   Apply this common eligibility rule to all six models.
4. Within training targets, use five expanding-window splits
   (TimeSeriesSplit, gap=0). Fit on the past and validate on the next block.
   Select table-model candidates by mean validation MSE, never test scores.
5. Refit the selected configuration on all training targets and evaluate the
   common later-period holdout. Recalculate MAE, RMSE, MSE and R² from saved
   predictions.

For the author's 336-row table, the raw cut is 268/68 records; after window
filtering, all models share **222 training targets and 68 test targets**.
Fold training sizes are 37, 74, 111, 148 and 185, with 37 validation targets
each. Missing hours mean equal counts do not imply equal calendar durations.
These counts are dataset-specific. This reference holdout was previously used
for the 15-predictor comparison; it is not a new external validation dataset.

LSTM uses X[t-23:t+1] to estimate y[t]. Current-hour environmental predictors
are observed, so this is **later-period contemporaneous regression**, not a
24-hour-ahead forecast. Scalers fit only each fold's training information.
LSTM uses one fixed configuration and 50 epochs; the final test period is not
used for early stopping. OLS also has one candidate.

Default XGBoost uses the original three-candidate grid. The optional
--extended-search produces a separately named XGBoost_extended model; it
does not silently replace the main model. Negative R² values are retained,
and no model is forced to rank first.

## SHAP, PDP and figures

- Explanations default to XGBoost and also support OLS, RF, CatBoost and
  LightGBM. Table explainers are not applied directly to the LSTM tensor.
- Every training target is explained. XGBoost/LightGBM use interventional
  TreeSHAP with all training targets as background; OLS uses LinearExplainer;
  RF/CatBoost use path-dependent TreeSHAP with training-tree cover counts.
  Metadata records these different assumptions. An independent additivity
  check compares SHAP sums plus the baseline with native predictions.
- Summary plots show all 13 variables. Dependence plots and 1D PDP default
  to the top six; --top-n 13 shows all variables across pages, and --features
  selects explicit variables.
- 2D PDP selects the first three training-SHAP-ranked variables. Each is
  paired with the three immediately following it: ranks (1,2–4), (2,3–5)
  and (3,4–6), giving nine distinct pairs.
- PDP averages predictions over **all training-background rows** after
  replacing the selected input(s). Continuous grids default to 31 points
  across training 5th–95th percentiles, or observed values when fewer exist.
  Weekday/hour grids use observed integer codes. Nine surfaces share a
  numerical colour scale derived from their actual prediction range.
- Figures export 600 dpi PNG and editable-text PDF/SVG, with alignment,
  font-size and collision checks. The prediction comparison has shared
  axis labels and one legend.

SHAP dependence shows attribution; PDP shows average model predictions.
Neither establishes causation or statistically significant interaction.
Correlated inputs can produce poorly supported PDP combinations. The
original iid bootstrap/LOWESS confidence bands are not carried forward.

## Outputs and saved fits

Outputs live under results/: protocol.json, models/, training/model/target/,
explanations/model/target/, summary/ and figures/. Source CSVs are not
overwritten. Data/model hashes guard against mixing incompatible results.

To reuse a matching **previously fitted 13-predictor** author analysis:

~~~bash
python prepare_data.py
python import_saved_results.py --source-results /path/to/paper5_no_mobility
python shap_analysis.py --targets EGE
python pdp_2d.py --targets EGE
~~~

The importer expects the previous author results format and checks data,
features, temporal indices and saved table-model predictions. It does not
convert old 15-predictor fits into 13-predictor fits.

## Validation scope

This update supplies revised code. It does not claim a new six-model training
run or new manuscript performance results. See [validation_report.json](validation_report.json)
for the checks actually run. Please cite the associated publication when
using this research workflow.
