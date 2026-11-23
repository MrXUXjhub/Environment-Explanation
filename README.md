# XGBoost-Based Explainable Analysis of Greenspace Exposure During Bike-Sharing Travel

[中文说明](README.zh-CN.md)

## Overview

This repository contains the source code used to model and interpret experienced greenspace exposure during bike-sharing travel using **XGBoost** and **Explainable Artificial Intelligence (XAI)** techniques.

The workflow also includes OLS, Random Forest, CatBoost, LightGBM and LSTM for model comparison. Each model has a separate Python script and uses a common data preparation and temporal validation procedure.

The workflow includes:

- Model training and hyperparameter selection
- Model performance evaluation
- SHAP feature importance analysis
- One-dimensional SHAP dependence and partial dependence analysis
- Two-dimensional Partial Dependence Plot (PDP) analysis

The code accompanies the research presented in the associated journal manuscript.

---

## Study Objective

This study investigates the relationships of temporal patterns, air pollution and weather conditions with experienced greenspace exposure (EGE) during bike-sharing travel.

The proposed explainable machine learning framework aims to:

- Predict greenspace exposure and its spatial distribution;
- Quantify the contributions of temporal, environmental and meteorological variables;
- Examine nonlinear relationships and joint model responses using SHAP and PDP analyses.

---

## Data Sources

The analyses integrate several publicly available datasets.

### Bike-Sharing Trip Data

The bike-sharing trip data were retrieved from the Mobike competition dataset provided through the Biendata platform:

https://www.biendata.xyz/

### Greenspace Data

Greenspace information was derived from the ESA WorldCover dataset:

https://esa-worldcover.org/en

### Air Pollution Data

Air pollution observations were obtained from the China National Environmental Monitoring Centre (CNEMC):

https://www.cnemc.cn/

### Meteorological Data

Meteorological variables were obtained from the China Meteorological Data Service Centre (CMDC):

https://data.cma.cn/

Prepare the joined hourly modelling table locally as `data/GE.csv`. Required columns and formatting are described in [data/README.md](data/README.md).

---

## Input Variables

### Predictor Variables

- Day of Week
- Hour of Day
- O3
- PM2.5
- PM10
- SO2
- NO2
- CO
- Temperature
- Wind Speed
- Pressure
- Humidity
- Precipitation

### Target Variables

The four outcomes are modelled separately:

- `SGE`: Origin greenspace exposure
- `EGE`: Destination greenspace exposure
- `GiniStart`: Gini coefficient of origin greenspace exposure
- `GiniEnd`: Gini coefficient of destination greenspace exposure

The predictor names and order are defined in `config.py`.

---

## Repository Structure

```text
.
├── data/
│   └── README.md
├── prepare_data.py
├── train_ols.py
├── train_rf.py
├── train_catboost.py
├── train_lightgbm.py
├── train_xgboost.py
├── train_lstm.py
├── shap_analysis.py
├── shap_dependence.py
├── pdp_1d.py
├── pdp_2d.py
├── summarize_results.py
├── plot_predictions.py
├── import_saved_results.py
├── run_all.py
├── self_check.py
├── config.py / runtime.py / data_protocol.py
├── training_common.py / explanation_common.py / plot_utils.py
├── tests/
├── figure_qa/
├── legacy/
├── README.md
├── README.zh-CN.md
├── requirements.txt
└── requirements_lstm.txt
```

Keep the shared modules and directory structure together. Models, predictions, metrics and figures are saved under `results/` when the scripts run.

---

## Software Requirements

The code uses **Python 3.12**.

Required packages include:

- numpy, pandas and scipy
- matplotlib and Pillow
- scikit-learn and joblib
- xgboost, lightgbm and catboost
- shap
- PyMuPDF
- tensorflow and keras for LSTM

Install the complete environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_lstm.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. For table models and their explanations, install `requirements.txt`. Package versions are recorded in `environment_versions.json`.

---

## Methodology

### 1. Model Training

- Prepare a common hourly modelling table.
- Reserve the final 20% of chronological records as the test period.
- Use five-fold expanding-window cross-validation within the training period.
- Select candidate configurations by mean validation MSE.
- Fit the selected configuration on the training targets and evaluate the common test targets.

All six models use the same eligible target hours. Each target requires 24 continuous input hours. LSTM uses the current and preceding 23 hours to estimate the current target, with scaling fitted only on training information. This evaluates contemporaneous regression in a later period. OLS and LSTM use one configuration each; the other models evaluate their specified candidate grids.

### 2. Model Evaluation

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- Coefficient of Determination (R²)

Metrics are recalculated from saved predictions. Actual-versus-predicted plots compare the six models with shared axis labels and one legend.

### 3. SHAP Analysis

SHAP is used to:

- Calculate feature importance
- Interpret feature contributions
- Explore nonlinear model responses
- Generate SHAP summary and dependence plots

Explanations default to XGBoost and use training-period observations. The scripts also support OLS and the other tree models. SHAP settings and additivity checks are saved with the outputs.

### 4. One-Dimensional SHAP Dependence Analysis

SHAP dependence plots show how feature values relate to their model attributions. Separate one-dimensional PDPs show average model predictions across a feature grid. The six highest-ranked variables are displayed by default; `--top-n 13` displays all predictors across pages.

### 5. Two-Dimensional Partial Dependence Analysis

Two-dimensional PDPs examine joint model responses to pairs of predictors. The three leading SHAP-ranked variables are each paired with the next three variables in the ranking, producing nine panels in a 3 × 3 layout.

Predictions are averaged over all training-background rows at each grid point. The panels share a numerical colour scale based on their prediction range. Figures are exported as 600 dpi PNG and editable-text PDF/SVG.

SHAP and PDP describe fitted-model behaviour; they do not by themselves establish causal effects or statistically significant interactions.

---

## Running the Code

Place the local modelling table at `data/GE.csv`, then run the complete workflow:

```bash
python run_all.py --stage all
```

The script will:

1. Prepare the modelling table and temporal splits.
2. Train and evaluate all six models for the four outcomes.
3. Summarize prediction performance.
4. Generate XGBoost SHAP feature importance and dependence plots.
5. Generate one- and two-dimensional PDPs.
6. Export actual-versus-predicted comparison plots.

To train and interpret XGBoost for EGE separately:

```bash
python prepare_data.py
python train_xgboost.py --targets EGE
python shap_analysis.py --targets EGE
python shap_dependence.py --targets EGE
python pdp_1d.py --targets EGE
python pdp_2d.py --targets EGE
```

Other models are run through their corresponding `train_*.py` files. Use `--model` to choose a table model for explanation and `--results` to specify an output directory. Run `--help` on an entry point for its options.

---

## Reproducibility

The repository provides the workflow for model fitting, evaluation and interpretation, subject to access to the prepared hourly data from the corresponding sources.

Please organize the local input table according to [data/README.md](data/README.md). Training saves fitted models under `results/models/`. The outputs record data fingerprints, temporal splits, model settings and row-level predictions.

Run the protocol and calculation checks without training real models:

```bash
python self_check.py
```

These checks use generated test data and also run through GitHub Actions. Matching saved author-model results can be imported with `import_saved_results.py`. Detailed commands are available in [使用说明.txt](使用说明.txt).

---

## Citation

If you use this repository in your research, please cite the associated publication.

---

## License

This repository is provided for academic research and educational purposes. Please cite the associated publication when using this code.
