"""解释器的输入、SHAP排序及真正的平均预测PDP。"""
from runtime import configure
configure()
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import FEATURES
from data_protocol import sha256
from training_common import load_tabular_model


def explanation_folder(bundle, model, target):
    return bundle.root/'explanations'/model/target


def load_explanation(bundle, model, target):
    estimator, model_meta = load_tabular_model(bundle, target, model)
    out = explanation_folder(bundle, model, target)
    path = out/'shap_metadata.json'
    if not path.is_file():
        raise FileNotFoundError('请先运行shap_analysis.py，生成该模型/目标的训练期SHAP排序。')
    meta = json.loads(path.read_text(encoding='utf-8'))
    if meta['protocol_sha256'] != bundle.protocol_hash or meta['model_sha256'] != model_meta['model_sha256']:
        raise ValueError('SHAP结果与当前数据/模型不匹配。')
    importance = pd.read_csv(out/'shap_importance.csv', encoding='utf-8-sig')
    values = pd.read_csv(out/'shap_values.csv', encoding='utf-8-sig')
    if sha256(out/'shap_values.csv') != meta['shap_values_sha256'] or sha256(out/'shap_importance.csv') != meta['importance_sha256']:
        raise ValueError('SHAP数值或重要性排序文件已改变。')
    ids = bundle.protocol['train_ids']
    if values.source_row.tolist() != ids:
        raise ValueError('SHAP没有覆盖相同的全部训练目标。')
    assert sorted(importance.feature.tolist()) == sorted(FEATURES)
    assert set(importance['rank'].tolist()) == set(range(1, 14))
    return estimator, bundle.X[ids], values[FEATURES].to_numpy(float), importance.sort_values('rank'), out


def choose_features(importance, explicit=None, top_n=6):
    selected = list(explicit) if explicit else importance.sort_values('rank').feature.tolist()[:top_n]
    if not selected or len(selected) != len(set(selected)) or not set(selected).issubset(FEATURES):
        raise ValueError('特征列表为空、重复或不在13变量白名单中。')
    return selected


def ranked_pairs(importance):
    order = importance.sort_values('rank').feature.tolist()
    pairs = [(order[r], order[r+c+1]) for r in range(3) for c in range(3)]
    assert len(pairs) == len(set(pairs)) == 9
    return pairs


def feature_grid(X, feature, resolution=31):
    if resolution < 2:
        raise ValueError('grid-resolution至少为2。')
    values = np.asarray(X)[:, FEATURES.index(feature)]
    unique = np.unique(values)
    if feature in ['Day of Week', 'Hour of Day']:
        if not np.all(unique % 1 == 0):
            raise ValueError(f'{feature}包含非整数编码。')
        return unique
    if len(unique) <= resolution:
        return unique
    low, high = np.quantile(values, [.05, .95])
    return np.unique(np.linspace(low, high, resolution))


def pdp_1d(estimator, X, feature, grid):
    X = np.asarray(X, float)
    j, predictions = FEATURES.index(feature), []
    for value in grid:
        modified = X.copy()
        modified[:, j] = value
        predictions.append(float(np.asarray(estimator.predict(modified)).mean()))
    return np.array(predictions)


def pdp_2d(estimator, X, feature_x, feature_y, grid_x, grid_y):
    X = np.asarray(X, float)
    j, k = FEATURES.index(feature_x), FEATURES.index(feature_y)
    surface = np.empty((len(grid_y), len(grid_x)))
    for row, value_y in enumerate(grid_y):
        for col, value_x in enumerate(grid_x):
            modified = X.copy()
            modified[:, j], modified[:, k] = value_x, value_y
            surface[row, col] = float(np.asarray(estimator.predict(modified)).mean())
    if not np.isfinite(surface).all():
        raise ValueError('PDP包含非有限值。')
    return surface
