# -*- coding: utf-8 -*-
########## XGBoost model training
# Basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Data splitting and cross-validation
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV

# Model evaluation
from sklearn.metrics import (
    mean_squared_error,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)
# Linear models
from sklearn.linear_model import LinearRegression, LogisticRegression

# Boosting models
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
# Model interpretation and visualization
from sklearn.inspection import PartialDependenceDisplay
from mlxtend.plotting import plot_decision_regions
import xgboost as xgb
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess

# Set the global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = 15
# Ensure minus signs are displayed correctly
plt.rcParams['axes.unicode_minus'] = False

# Load the dataset and define predictor and target variables
data=pd.read_csv(r'E:\data\csv\GE.csv')
X = data[['Day of Week', 'Hour of Day', 'O3','PM2.5','PM10','SO2','NO2','CO','Temperature','Wind Speed','Pressure','Humidity','Precipitation', 'Trip Volume','Travel Distance']]# Predictor variables
y = data['EGE']# Target variable

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# XGBoost regression parameters
params_xgb = {
    'learning_rate': 0.02,            # Learning rate
    'booster': 'gbtree',              # Booster type
    'objective': 'reg:squarederror',  # Regression objective
    'max_leaves': 127,                # Maximum number of leaf nodes
    'verbosity': 1,                   # Verbosity level
    'seed': 42,                       # Random seed
    'nthread': -1,                    # Number of CPU threads
    'colsample_bytree': 0.6,          # Feature sampling ratio
    'subsample': 0.7,                 # Sample sampling ratio
    'eval_metric': 'rmse'             # Evaluation metric (RMSE)
 }
# Initialize the XGBoost regression model
model_xgb = xgb.XGBRegressor(**params_xgb)
# Define the parameter grid for hyperparameter tuning
param_grid = {
    # 'n_estimators': [900, 1000,1100,1200,1300,1400],  # Number of trees
    'n_estimators': [600,800,1000],  # Number of trees
    'max_depth': [3],                      # Maximum tree depth
    'learning_rate': [ 0.02],   # Learning rate
 }
# Perform grid search with five-fold cross-validation
grid_search = GridSearchCV(
    estimator=model_xgb,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Evaluation metric
    cv=5,                              # Five-fold cross-validation
    n_jobs=-1,                         # Parallel computing
    verbose=1                          # Display training progress
 )
# Train the model
grid_search.fit(X_train, y_train)
# Print the optimal hyperparameters
print("Best parameters found: ", grid_search.best_params_)
# print("Best RMSE score: ", (-grid_search.best_score_) ** 0.5)  # Convert back to RMSE
# Retrieve the best-performing model
best_model_xgboost = grid_search.best_estimator_



# Generate predictions and evaluate model performance
y_train_pred = best_model_xgboost.predict(X_train)# Predict the target values
y_pred = best_model_xgboost.predict(X_test)# Predict the target values
Train_MAE = mean_absolute_error(y_train, y_train_pred)
Train_MSE = mean_squared_error(y_train, y_train_pred)
Train_RMSE = np.power(Train_MSE,0.5)
Train_R2 = r2_score(y_train, y_train_pred)

Test_MAE = mean_absolute_error(y_test, y_pred)
Test_MSE = mean_squared_error(y_test, y_pred)
Test_RMSE = np.power(Test_MSE,0.5)
Test_R2 = r2_score(y_test, y_pred)

# Evaluate model performance on the training set
print("Train MAE:",Train_MAE)# Mean Absolute Error
# print("Train MSE:",Train_MSE)# Mean Squared Error
print("Train RMSE:",Train_RMSE)# Mean Squared Error
print("Train R2:",Train_R2)# Coefficient of determination (R²)
# Evaluate model performance on the test set
print("Test MAE:",Test_MAE)# Mean Absolute Error
# print("Test MSE:",Test_MSE)# Mean Squared Error
print("Test RMSE:",Test_RMSE)# Mean Squared Error
print("Test R2:",Test_R2)# Coefficient of determination (R²)

# Plot actual versus predicted values
plt.figure(figsize=(6,6),dpi=350)# Set figure size
plt.scatter(y_train, y_train_pred, color="#1f77b4", label='Train', alpha=0.8)
plt.scatter(y_test, y_pred, color="#d62728", label='Test', alpha=0.8)

plt.plot([0.061, 0.23], [0.061, 0.23], color='black', linestyle='--', label='y = x')
plt.xlim(0.061, 0.23)  # Set x-axis range
plt.ylim(0.061, 0.23)  # Set y-axis range
# Define axis tick intervals
x_ticks = np.arange(0.08, 0.24, 0.02)  # From 0.061 to 0.23 with an interval of 0.02
y_ticks = np.arange(0.08, 0.24, 0.02)
# Apply tick settings
plt.xticks(x_ticks)
plt.yticks(y_ticks)

# plt.plot([0.42, 0.7], [0.42, 0.7], color='black', linestyle='--', label='y = x')
# plt.xlim(0.42, 0.7)  # Set x-axis range
# plt.ylim(0.42, 0.7)  # Set y-axis range
# # Define axis tick intervals
# x_ticks = np.arange(0.45, 0.7, 0.05)  # From 0.061 to 0.23 with an interval of 0.02
# y_ticks = np.arange(0.45, 0.7, 0.05)
# # Apply tick settings
# plt.xticks(x_ticks)
# plt.yticks(y_ticks)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.savefig(r'E:\data\paper5\figure\XGBoostEvaluationEGE.jpg', bbox_inches='tight', dpi=350)
plt.show()

############################### SHAP analysis
# Create a SHAP explainer
explainer = shap.TreeExplainer(best_model_xgboost)
# Compute SHAP values
shap_values = explainer.shap_values(X)
# Feature names
labels = X.columns

# SHAP summary plot
# plt.figure(dpi=350)
# shap.summary_plot(shap_values, X, feature_names=labels, plot_type="dot", show=False, cmap='Spectral_r')    #'RdBu_r','Spectral_r'

# plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
# # Top and right spines
# plt.gca().spines['top'].set_visible(True)
# plt.gca().spines['left'].set_visible(True)
# plt.gca().spines['right'].set_visible(True)
# plt.xlim(-0.05, 0.02)  # Set x-axis range
# # plt.savefig(r'E:\data\paper5\figure\XGBoostSHAP点图1EGE.jpg', bbox_inches='tight', pad_inches=0.1, dpi=350)
# plt.show()

# Combined SHAP beeswarm and bar plot
# Nature-style color palette
colors = {
    'Time factors': '#4B6A88',        # Dark blue
    'Air pollution': '#D55E00',      # Brick red
    'Weather conditions': '#0072B2', # Dark green
    'Travel-related factors': '#CC79A7' # Purple
}

feature_colors = {feature: colors['Air pollution'] for feature in ['O3', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO']}
feature_colors.update({feature: colors['Travel-related factors'] for feature in ['Trip Volume', 'Travel Distance']})
feature_colors.update({feature: colors['Weather conditions'] for feature in ['Temperature', 'Wind Speed', 'Pressure', 'Humidity', 'Precipitation']})
feature_colors.update({feature: colors['Time factors'] for feature in ['Day of Week', 'Hour of Day']})

# Compute feature importance from SHAP values
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_names = X.columns

# Plot feature importance
sorted_idx = np.argsort(mean_abs_shap)  # Sort by feature importance
sorted_features = feature_names[sorted_idx]
sorted_colors = [feature_colors[f] for f in sorted_features]


# Create the main figure for the bar plot
fig, ax1 = plt.subplots(figsize=(10, 8), dpi=350)

# Plot feature importance bars
# shap.summary_plot(shap_values, X,feature_names=labels, plot_type="bar", show=False)
bars = plt.barh(sorted_features, mean_abs_shap[sorted_idx], color=sorted_colors)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # Adjust plot position
# Configure plot borders
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
# Adjust transparency
bars = ax1.patches  # Get all bar objects
for bar in bars:
    bar.set_alpha(0.2)  # Set transparency

# Get the shared y-axis
ax1 = plt.gca()

# Create a twin axis for the SHAP beeswarm plot
ax2 = ax1.twiny()
shap.summary_plot(shap_values, X, feature_names=labels, plot_type="dot", show=False, cmap='Spectral_r')
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # Align the plot with the bar chart
# plt.xlim(-0.05, 0.02)  # Set x-axis range

# Configure the labels for both x-axes
# Set the color of the top x-axis
ax1.spines['top'].set_color("#1f77b4")  # Set the top axis color
ax1.tick_params(axis='x', colors="#1f77b4", labelsize=12)  # Set the top-axis ticks and labels
ax1.tick_params(axis='y', labelsize=15)  # Set the top-axis ticks and labels
ax1.set_xlabel('Mean SHAP Value (Feature Importance)', fontsize=15, color="#1f77b4")  # Set the top-axis label color

# Move the top x-axis to avoid overlapping
ax1.xaxis.set_label_position('top')  # Move the label to the top
ax1.xaxis.tick_top()  # Move the ticks to the top

ax2.spines['bottom'].set_color("#d62728")  # Set the top axis color
ax2.tick_params(axis='x', colors="#d62728", labelsize=12)  # Set the top-axis ticks and labels

ax2.xaxis.set_label_position('bottom')  # Move the label to the top
ax2.set_xlabel('SHAP Value Contribution (Bee Swarm)', color="#d62728", fontsize=15)
plt.tight_layout()
plt.savefig(r'E:\data\paper5\figure\XGBoostSHAP1GiniEnd.jpg', bbox_inches='tight', dpi=350)
plt.show()


######################## One-dimensional SHAP dependence analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# Selected features
features = ['Hour of Day', 'Trip Volume', 'Temperature','Humidity',  'O3','Travel Distance',  'Day of Week',  'NO2']#EGE
scattercolor = "#1f77b4"
linecolor = "#d62728"
# Create a 3 × 3 subplot grid
fig, axs = plt.subplots(3, 3, figsize=(16, 12), dpi=350)

# Flatten the subplot array
axs = axs.flatten()

for i, feature_name in enumerate(features):
    feature_values = X[feature_name]
    shap_values_feature = shap_values[:, X.columns.get_loc(feature_name)]  # Get SHAP values for the current feature

    # Prepare data
    data = pd.DataFrame({
        'feature': feature_values,
        'shap_value': shap_values_feature
    })

    # Fit a LOWESS curve
    sorted_feature = np.sort(data['feature'].values)
    lowess_fit = lowess(data['shap_value'], data['feature'], frac=0.3)  # frac controls the smoothing level
    lowess_x = lowess_fit[:, 0]
    lowess_y = lowess_fit[:, 1]

    # Estimate the confidence interval using bootstrap
    bootstrap_y = []
    for j in range(100):  # Repeat bootstrap sampling 100 times
        sample = data.sample(frac=1, replace=True)
        lowess_sample = lowess(sample['shap_value'], sample['feature'], frac=0.3)
        bootstrap_y.append(lowess_sample[:, 1])
    bootstrap_y = np.array(bootstrap_y)
    ci_lower = np.percentile(bootstrap_y, 2.5, axis=0)  # 2.5th percentile
    ci_upper = np.percentile(bootstrap_y, 97.5, axis=0)  # 97.5th percentile

    # Select the corresponding subplot
    ax1 = axs[i]

    # Plot SHAP values with the fitted LOWESS curve
    ax1.scatter(data['feature'], data['shap_value'], color=scattercolor, s=20, alpha=0.9, label="Samples")
    ax1.plot(lowess_x, lowess_y, color='orange', linestyle='--', linewidth=2,alpha=0.95, label="Fitted Curve")
    ax1.fill_between(lowess_x, ci_lower, ci_upper, color='orange', alpha=0.2, label="95% Confidence Interval")

    ax1.set_xlabel(f"{feature_name}", fontsize=15)
    ax1.set_ylabel(f"SHAP Value", fontsize=15)
    # ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True)
    ax1.tick_params(axis='x', labelsize=12)  # Set x-axis tick labels
    ax1.tick_params(axis='y', labelsize=12)  # Set x-axis tick labels
    ax1.grid(which='major', linestyle='--', linewidth=0.5, color='lightgray')
    ax1.axhline(y=0, color=linecolor, linestyle='--', linewidth=2, alpha=0.9)

    # Plot the feature distribution
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)  # Hide the top spine
    ax2.spines['right'].set_visible(False)  # Hide the right spine
    ax2.spines['left'].set_visible(False)  # Hide the left spine

    # Plot histogram
    ax2.hist(data['feature'], bins=30, color='gray', alpha=0.7, orientation='vertical')
    plt.ylim(0, 120)  # Set y-axis range
    ax2.yaxis.set_ticks_position('none')  # Hide right y-axis ticks
    ax2.set_yticklabels([])  # Hide right y-axis tick labels

# Remove the unused subplot
fig.delaxes(axs[-1])

# Adjust layout
plt.tight_layout()
plt.savefig(r'E:\data\paper5\figure\SHAP_dependenceEGE.jpg', bbox_inches='tight', dpi=350)
plt.show()

##################### Two-dimensional PDP analysis
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colorbar import Colorbar
# Define all features
features = ['Hour of Day', 'Trip Volume', 'Temperature','Humidity',  'O3','Travel Distance',  'Day of Week',  'NO2',
            'PM2.5','Pressure','PM10','Precipitation','Wind Speed','SO2','CO'
]#EGE

# Select the first five features
first_five = features[:5]  # First five features

# Create a 5 × 5 subplot grid
fig, axes = plt.subplots(5, 5, figsize=(20, 20), dpi=350)
axes = axes.ravel()  # Flatten the subplot array

# Iterate over feature combinations
counter = 0  # Current subplot index
colorbar_axs = []  # Store the colorbar axis
for i, feature_1 in enumerate(first_five):
    next_five = features[i + 1: i + 6]  # Select the next five features
    for feature_2 in next_five:
        ax = axes[counter]  # Select subplot
        PartialDependenceDisplay.from_estimator(
            best_model_xgboost, X, [(feature_1, feature_2)], ax=ax
        )
        counter += 1

# Remove unused subplots
for k in range(counter, 25):
    fig.delaxes(axes[k])

# Add a shared colorbar
cmap = cm.viridis  # Use the viridis colormap
norm = plt.Normalize(vmin=-1, vmax=1)  # Set the color scale
cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Set the colorbar position
ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')

# Adjust subplot spacing
fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
# # Adjust layout
# plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space on the right for the colorbar
plt.savefig(r'E:\data\paper5\figure\2D_PDP_PlotEGE.jpg', bbox_inches='tight', dpi=350)
plt.show()






