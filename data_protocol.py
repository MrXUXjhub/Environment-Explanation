"""生成共同时间划分；所有模型只能读取13列白名单输入。"""
from __future__ import annotations
from runtime import configure
configure()
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from config import (FEATURES, TARGETS, EXCLUDED, DEFAULT_RESULTS,
                    SEQUENCE_LENGTH, N_SPLITS, AUTHOR_CONFIRMED_CSV_SHA256)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def prepare(csv_path, results=DEFAULT_RESULTS, year=None):
    csv_path, results = Path(csv_path).resolve(), Path(results).resolve()
    raw = pd.read_csv(csv_path, encoding='utf-8-sig')
    required = set(FEATURES + TARGETS)
    if not required.issubset(raw.columns):
        raise ValueError(f'CSV缺少列: {sorted(required-set(raw.columns))}')
    source_hash = sha256(csv_path)
    if 'timestamp' in raw.columns:
        times = pd.to_datetime(raw['timestamp'], errors='raise')
    elif 'date' in raw.columns:
        hours = pd.to_numeric(raw['Hour of Day'], errors='raise')
        if not ((hours >= 0) & (hours <= 23) & (hours % 1 == 0)).all():
            raise ValueError('Hour of Day必须为0—23的整数。')
        times = pd.to_datetime(raw['date'], errors='raise').dt.normalize() + pd.to_timedelta(hours, unit='h')
    else:
        raise ValueError('CSV需包含timestamp或date日期列。')
    source_years = sorted(int(x) for x in times.dt.year.unique())
    author_year_correction = (year is None and source_hash == AUTHOR_CONFIRMED_CSV_SHA256
                              and source_years == [2023])
    effective_year = 2017 if author_year_correction else year
    if effective_year is not None:
        times = times.map(lambda t: t.replace(year=int(effective_year)))
    if times.duplicated().any():
        raise ValueError('时间戳重复；请先核查数据，不能把重复小时当作独立小时。')
    if not np.array_equal(times.dt.hour.to_numpy(), raw['Hour of Day'].to_numpy()):
        raise ValueError('timestamp与Hour of Day不一致。')
    ordering = np.argsort(times.to_numpy(), kind='stable')
    times = times.iloc[ordering].reset_index(drop=True)
    frame = raw.iloc[ordering][FEATURES + TARGETS].reset_index(drop=True).apply(pd.to_numeric, errors='raise')
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError('模型列含缺失或无穷值；本脚本不会全样本插补。')
    frame.insert(0, 'timestamp', [t.isoformat() for t in times])
    frame.insert(0, 'source_row', range(len(frame)))
    eligible = [i for i in range(SEQUENCE_LENGTH-1, len(frame))
                if (times.iloc[i-SEQUENCE_LENGTH+1:i+1].diff().dropna()
                    == pd.Timedelta(hours=1)).all()]
    cut = int(len(frame) * .8)
    train, test = [i for i in eligible if i < cut], [i for i in eligible if i >= cut]
    if len(train) < N_SPLITS+1 or not test:
        raise ValueError('连续历史和训练/测试划分后的样本不足。')
    folds = []
    for k, (a, b) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS, gap=0).split(train), 1):
        tr, va = [train[int(i)] for i in a], [train[int(i)] for i in b]
        folds.append({'fold': k, 'train_ids': tr, 'validation_ids': va,
                      'train_start': frame.timestamp.iloc[tr[0]],
                      'train_end': frame.timestamp.iloc[tr[-1]],
                      'validation_start': frame.timestamp.iloc[va[0]],
                      'validation_end': frame.timestamp.iloc[va[-1]]})
    results.mkdir(parents=True, exist_ok=True)
    prepared = results / 'prepared_model_data.csv'
    frame.to_csv(prepared, index=False, encoding='utf-8-sig')
    pd.DataFrame({'source_row': range(len(frame)), 'original_csv_row': ordering}).to_csv(results/'row_mapping.csv', index=False)
    protocol = {'schema_version': 1, 'features': FEATURES, 'targets': TARGETS,
        'excluded_predictors': EXCLUDED, 'raw_csv_sha256': source_hash,
        'prepared_data_sha256': sha256(prepared), 'n_source_rows': len(frame),
        'source_years': source_years, 'effective_year_override': effective_year,
        'author_confirmed_year_correction': author_year_correction,
        'timestamps': frame.timestamp.tolist(), 'eligible_ids': eligible,
        'global_chronological_cut_index': cut, 'train_ids': train, 'test_ids': test,
        'cv_folds': folds, 'gap': 0, 'n_splits': N_SPLITS,
        'sequence_length': SEQUENCE_LENGTH, 'sequence_includes_current_X': True,
        'excluded_target_rows': [i for i in range(len(frame)) if i not in eligible],
        'selection': 'Mean validation MSE across five expanding-window folds; never choose candidates by held-out test scores.',
        'task': 'Later-period contemporaneous regression; current environmental predictors are observed, not a 24-hour-ahead forecast.',
        'calendar_note': 'Equal validation counts do not imply equal calendar durations when hours are missing.',
        'holdout_note': 'For the supplied dataset this is the same reference holdout used in the previous 15-feature comparison, not new external validation.'}
    write_json(results/'protocol.json', protocol)
    return load_bundle(results)


@dataclass
class DataBundle:
    root: Path
    frame: pd.DataFrame
    protocol: dict
    protocol_hash: str

    @property
    def X(self):
        return self.frame[FEATURES].to_numpy(dtype=float)

    def y(self, target):
        if target not in TARGETS:
            raise ValueError(f'未知目标: {target}')
        return self.frame[target].to_numpy(dtype=float)


def load_bundle(results=DEFAULT_RESULTS):
    root = Path(results).resolve()
    protocol_path = root/'protocol.json'
    if not protocol_path.is_file():
        raise FileNotFoundError('请先执行prepare_data.py。')
    p = json.loads(protocol_path.read_text(encoding='utf-8'))
    if p['features'] != FEATURES or len(FEATURES) != 13:
        raise ValueError('输入顺序或13变量白名单已改变。')
    path = root/'prepared_model_data.csv'
    if sha256(path) != p['prepared_data_sha256']:
        raise ValueError('建模表已改变；需重新准备并使用匹配的模型。')
    d = pd.read_csv(path, encoding='utf-8-sig')
    if list(d.columns) != ['source_row', 'timestamp', *FEATURES, *TARGETS]:
        raise ValueError('建模表包含额外列或缺少列。')
    assert set(EXCLUDED).isdisjoint(d.columns)
    assert d.source_row.tolist() == list(range(len(d)))
    assert d.timestamp.tolist() == p['timestamps']
    assert set(p['train_ids']).isdisjoint(p['test_ids'])
    assert max(p['train_ids']) < min(p['test_ids'])
    assert sorted(p['train_ids'] + p['test_ids']) == p['eligible_ids']
    for f in p['cv_folds']:
        assert max(f['train_ids']) < min(f['validation_ids'])
        assert set(f['train_ids'] + f['validation_ids']).issubset(p['train_ids'])
    return DataBundle(root, d, p, sha256(protocol_path))


def add_common_arguments(parser, *, model=False):
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS, help='共同数据、模型与结果目录')
    parser.add_argument('--targets', nargs='+', choices=TARGETS, default=TARGETS)
    if model:
        from config import TABULAR_MODELS
        parser.add_argument('--model', choices=TABULAR_MODELS, default='XGBoost')
    return parser
