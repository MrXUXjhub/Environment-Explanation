# XGBoost-Based Explainable Analysis of Greenspace Exposure During Bike-Sharing Travel

[中文说明](README.zh-CN.md)

## Overview

This repository contains the source code used to model and interpret experienced greenspace exposure during bike-sharing travel using **XGBoost** and **Explainable Artificial Intelligence (XAI)** techniques.

The workflow also includes OLS, Random Forest, CatBoost, LightGBM and LSTM for model comparison. Each model has a standalone Python file. SHAP and PDP are combined in one file. Data paths, target variables and model parameters are set at the top of each script.

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

The predictor names and order are listed in `FEATURES` at the top of each script.

---

## Repository Structure

```text
.
├── OLS.py
├── RF.py
├── CatBoost.py
├── LightGBM.py
├── XGBoost.py
├── LSTM.py
├── SHAP_PDP.py
├── data/
│   └── README.md
├── README.md
├── README.zh-CN.md
└── requirements.txt
```

Place the local hourly table at `data/GE.csv`. Each Python file runs independently; it does not import another script from this repository. Model outputs are saved under `results/<model>/`, and explanations under `results/SHAP_PDP/<model>/`.

---

## Software Requirements

The code uses **Python 3.12**.

Required packages include:

- numpy, pandas and scipy
- matplotlib and Pillow
- scikit-learn and joblib
- xgboost, lightgbm and catboost
- shap
- tensorflow and keras for LSTM

Install the complete environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. Package versions are pinned in `requirements.txt`. TensorFlow and Keras are needed for LSTM.

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

SHAP dependence plots show how feature values relate to their model attributions. Separate one-dimensional PDPs show average model predictions across a feature grid. The six highest-ranked variables are displayed by default; set `TOP_N = 13` in `SHAP_PDP.py` to display all predictors across pages.

### 5. Two-Dimensional Partial Dependence Analysis

Two-dimensional PDPs examine joint model responses to pairs of predictors. The three leading SHAP-ranked variables are each paired with the next three variables in the ranking, producing nine panels in a 3 × 3 layout.

Predictions are averaged over all training-background rows at each grid point. The panels share a numerical colour scale based on their prediction range. Figures are exported as 600 dpi PNG and editable-text PDF/SVG.

---

## Running the Code

Place the local modelling table at `data/GE.csv`. Run the model you need directly:

```bash
python OLS.py
python RF.py
python CatBoost.py
python LightGBM.py
python XGBoost.py
python LSTM.py
```

Each model script prepares its data, performs temporal validation, fits the model and exports metrics, predictions and an actual-versus-predicted figure. Edit `DATA_FILE`, `TARGETS`, `OUTPUT_DIR` and the model settings at the top of the file as needed.

To generate SHAP and PDP analyses for XGBoost, first run `XGBoost.py`, then:

```bash
python SHAP_PDP.py
```

This file loads the saved model and its training background, then generates the SHAP summary, SHAP dependence plots, one-dimensional PDPs and the 3 × 3 two-dimensional PDP figure. It does not refit the model.

Set `MODEL_NAME` in `SHAP_PDP.py` to select a fitted table model, and edit `TARGETS`, `TOP_N` or `GRID_POINTS` for the desired outputs. The explanation file supports OLS, RF, CatBoost, LightGBM and XGBoost.

---

## Reproducibility

The repository provides the workflow for model fitting, evaluation and interpretation, subject to access to the prepared hourly data from the corresponding sources.

Input formatting is described in [data/README.md](data/README.md). All model scripts use the same predictor order and chronological splitting rules. Saved outputs include metrics, row-level predictions, cross-validation results and fitted models. Table-model files also contain the training background used for SHAP and PDP.

To check the input and eligible sample counts without fitting a model:

```bash
python XGBoost.py --check-only
```

GitHub Actions checks the syntax of the seven Python files.

---

## Citation

If you use this repository in your research, please cite the associated publication.

---

## License

This repository is provided for academic research and educational purposes. Please cite the associated publication when using this code.
