"""OLS和树模型共用的真实前向验证、保存模型及逐行精度计算。"""
from __future__ import annotations
from runtime import configure
configure()
import argparse
import json
import platform
import time
from pathlib import Path
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import FEATURES, TARGETS
from data_protocol import add_common_arguments, load_bundle, sha256, write_json


def scores(y, predicted):
    y, predicted = np.asarray(y, float).ravel(), np.asarray(predicted, float).ravel()
    if y.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError('预测维度或有限性检查失败。')
    mse = float(mean_squared_error(y, predicted))
    result = {'n': len(y), 'MAE': float(mean_absolute_error(y, predicted)),
              'RMSE': float(np.sqrt(mse)), 'MSE': mse,
              'R2': float(r2_score(y, predicted))}
    assert result['MAE'] <= result['RMSE'] + 1e-12
    return result


def artifact_paths(root, target, model):
    folder = Path(root)/'models'
    return folder/f'{target}__{model}.joblib', folder/f'{target}__{model}.json'


def check_estimator_feature_count(estimator):
    count = getattr(estimator, 'n_features_in_', None)
    # 部分CatBoost序列化版本将sklearn包装器计数恢复为0，原生特征名仍完整。
    if (count is None or count == 0) and type(estimator).__name__ == 'CatBoostRegressor':
        count = len(estimator.feature_names_)
    if count != len(FEATURES):
        raise ValueError(f'已拟合模型需要{count}个输入，预期为13。')


def save_tabular_model(bundle, target, name, estimator, params, cv_mse, **extra):
    path, meta_path = artifact_paths(bundle.root, target, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    check_estimator_feature_count(estimator)
    joblib.dump(estimator, path)
    metadata = {'target': target, 'model': name, 'features': FEATURES,
        'n_features': 13, 'protocol_sha256': bundle.protocol_hash,
        'model_sha256': sha256(path), 'parameters': params,
        'mean_validation_mse': float(cv_mse), 'CV_RMSE': float(np.sqrt(cv_mse)),
        'training_rows': bundle.protocol['train_ids'],
        'test_rows': bundle.protocol['test_ids'], **extra}
    write_json(meta_path, metadata)
    return metadata


def load_tabular_model(bundle, target, name):
    path, meta_path = artifact_paths(bundle.root, target, name)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f'缺少{target}/{name}已拟合模型；先运行训练或导入脚本。')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta['protocol_sha256'] != bundle.protocol_hash or meta['features'] != FEATURES:
        raise ValueError('模型与共同数据/输入顺序不匹配。')
    if meta['target'] != target or meta['model'] != name or meta['model_sha256'] != sha256(path):
        raise ValueError('模型文件或元数据校验失败。')
    estimator = joblib.load(path)
    check_estimator_feature_count(estimator)
    return estimator, meta


def cross_validate(bundle, target, factory, candidates):
    X, y = bundle.X, bundle.y(target)
    rows, best = [], None
    for c, params in enumerate(candidates):
        mses = []
        for f in bundle.protocol['cv_folds']:
            tr, va = f['train_ids'], f['validation_ids']
            estimator = factory(params)
            estimator.fit(X[tr], y[tr])
            score = scores(y[va], estimator.predict(X[va]))
            mses.append(score['MSE'])
            rows.append({'candidate': c, 'fold': f['fold'], **score,
                         'train_n': len(tr), 'train_end': f['train_end'],
                         'validation_start': f['validation_start'],
                         'parameters': json.dumps(params, sort_keys=True)})
        mean_mse = float(np.mean(mses))
        if best is None or mean_mse < best['mean_validation_mse']:
            best = {'candidate': c, 'params': params, 'mean_validation_mse': mean_mse}
    return pd.DataFrame(rows), best


def prediction_records(bundle, target, name, train_predictions, test_predictions):
    rows, metrics = [], []
    y = bundle.y(target)
    for split, predictions in [('train', train_predictions), ('test', test_predictions)]:
        ids = bundle.protocol['train_ids' if split == 'train' else 'test_ids']
        prediction = np.asarray(predictions, dtype=float).ravel()
        metrics.append({'target': target, 'model': name, 'split': split, **scores(y[ids], prediction)})
        for i, yp in zip(ids, prediction):
            rows.append({'target': target, 'model': name, 'split': split,
                         'source_row': i, 'timestamp': bundle.protocol['timestamps'][i],
                         'observed': float(y[i]), 'predicted': float(yp)})
    return pd.DataFrame(rows), pd.DataFrame(metrics)


def fit_tabular(bundle, target, name, factory, candidates):
    start = time.monotonic()
    out = bundle.root/'training'/name/target
    out.mkdir(parents=True, exist_ok=True)
    # 写出完整候选列表，测试精度不会改变候选或排序。
    write_json(out/'candidates_frozen.json', candidates)
    cv, best = cross_validate(bundle, target, factory, candidates)
    cv.to_csv(out/'cross_validation.csv', index=False)
    estimator = factory(best['params'])
    X, y = bundle.X, bundle.y(target)
    tr, te = bundle.protocol['train_ids'], bundle.protocol['test_ids']
    estimator.fit(X[tr], y[tr])
    metadata = save_tabular_model(bundle, target, name, estimator, best['params'],
                                 best['mean_validation_mse'], selected_candidate=best['candidate'],
                                 candidate_count=len(candidates),
                                 sklearn_version=importlib.metadata.version('scikit-learn'),
                                 python_version=platform.python_version(),
                                 elapsed_seconds=time.monotonic()-start)
    pred, metrics = prediction_records(bundle, target, name,
                                      estimator.predict(X[tr]), estimator.predict(X[te]))
    metrics['CV_RMSE'] = metadata['CV_RMSE']
    pred.to_csv(out/'predictions.csv', index=False)
    metrics.to_csv(out/'metrics.csv', index=False)
    print(f'{name} {target}: training CV_RMSE={metadata["CV_RMSE"]:.6f}; final test scores saved.', flush=True)


def model_cli(name, factory, grid, argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=f'{name}: 13 predictors, five expanding-window folds.'))
    parser.add_argument('--check-only', action='store_true', help='只检查数据和候选数量，不拟合模型')
    args = parser.parse_args(argv)
    bundle = load_bundle(args.results)
    candidates = list(ParameterGrid(grid))
    if args.check_only:
        print(f'{name}: {len(candidates)} candidate(s), 5 folds, 13 predictors; no model fitted.')
        return
    for target in args.targets:
        fit_tabular(bundle, target, name, factory, candidates)
