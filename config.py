"""所有训练和解释脚本共同使用的变量、目标和路径。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / 'data' / 'GE.csv'
DEFAULT_RESULTS = ROOT / 'results'
FEATURES = ['Day of Week', 'Hour of Day', 'O3', 'PM2.5', 'PM10', 'SO2',
            'NO2', 'CO', 'Temperature', 'Wind Speed', 'Pressure', 'Humidity',
            'Precipitation']
TARGETS = ['SGE', 'EGE', 'GiniStart', 'GiniEnd']
EXCLUDED = ['ordercount', 'Trip Volume', 'Ln_Trip Volume', 'Travel Distance']
TREE_MODELS = ['RF', 'CatBoost', 'LightGBM', 'XGBoost', 'XGBoost_extended']
TABULAR_MODELS = ['OLS', *TREE_MODELS]
MAIN_MODELS = ['OLS', 'RF', 'CatBoost', 'LightGBM', 'XGBoost', 'LSTM']
SEED, LSTM_SEED, N_JOBS, N_SPLITS, SEQUENCE_LENGTH = 42, 5, 2, 5, 24
DISPLAY = {'Day of Week': 'Day of week', 'Hour of Day': 'Hour of day',
           'Wind Speed': 'Wind speed', 'O3': 'O₃', 'NO2': 'NO₂',
           'SO2': 'SO₂', 'PM2.5': 'PM₂.₅', 'PM10': 'PM₁₀'}
TARGET_LABELS = {'SGE': 'Origin GE', 'EGE': 'Destination GE',
                 'GiniStart': 'Origin Gini', 'GiniEnd': 'Destination Gini'}
# 仅对作者已确认的这份CSV应用2023 -> 2017修正；新数据不自动改年份。
AUTHOR_CONFIRMED_CSV_SHA256 = '7ef31080ed421e61560c292fe60866da8b61fec013bfa6235111c0dddf8cee0b'
