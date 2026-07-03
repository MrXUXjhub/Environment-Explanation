# XGBoost-Based Explainable Analysis of Greenspace Exposure During Bike-Sharing Travel

## Overview

This repository contains the source code used to model and interpret experienced greenspace exposure during bike-sharing travel using **XGBoost** and **Explainable Artificial Intelligence (XAI)** techniques.

The workflow includes:

- XGBoost model training and hyperparameter optimization
- Model performance evaluation
- SHAP feature importance analysis
- One-dimensional SHAP dependence analysis with LOWESS smoothing
- Two-dimensional Partial Dependence Plot (PDP) analysis

The code accompanies the research presented in the associated journal manuscript.

---

## Study Objective

This study investigates how travel behavior, air pollution, and weather conditions jointly influence experienced greenspace exposure (EGE) during bike-sharing travel.

The proposed explainable machine learning framework aims to:

- Predict experienced greenspace exposure;
- Quantify the contribution of travel, environmental, and meteorological variables;
- Reveal nonlinear relationships and interaction effects using SHAP and PDP analyses.

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
- Trip Volume
- Travel Distance

### Target Variable

- Experienced Greenspace Exposure

---

## Repository Structure

```text
.
├── data/
│   └── GE.csv
├── figures/
├── Environment_Explanation.py
├── README.md
└── requirements.txt
```

---

## Software Requirements

The code was developed in Python 3.

Required packages include:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- shap
- statsmodels
- mlxtend

## Methodology

### 1. XGBoost Model Training

- Data preprocessing
- Train/test split
- Hyperparameter tuning using GridSearchCV
- Five-fold cross-validation

### 2. Model Evaluation

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### 3. SHAP Analysis

SHAP is used to:

- Calculate feature importance
- Interpret feature contributions
- Explore nonlinear effects
- Generate SHAP summary and dependence plots

### 4. One-Dimensional SHAP Dependence Analysis

LOWESS smoothing is applied to visualize nonlinear relationships and confidence intervals.

### 5. Two-Dimensional Partial Dependence Analysis

Two-dimensional PDPs are used to investigate interaction effects between predictor variables.

---

## Running the Code

Run the main script:

```bash
python Environment_Explanation.py
```

The script will:

1. Train the XGBoost model.
2. Evaluate prediction performance.
3. Generate SHAP feature importance plots.
4. Generate one-dimensional SHAP dependence plots.
5. Generate two-dimensional PDP visualizations.

---

## Reproducibility

All analyses presented in the associated manuscript can be reproduced using the code provided in this repository, subject to access to the original datasets from the corresponding data providers.

Please organize the input datasets according to the directory structure expected by the script before execution.

---

## Citation

If you use this repository in your research, please cite the associated publication.

---

## License

This repository is provided for academic research and educational purposes. Please cite the associated publication when using this code.
