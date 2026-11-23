"""XGBoost：默认保持原始三组网格；扩展搜索须显式启用并单独保存。"""
import argparse
from sklearn.model_selection import ParameterGrid, ParameterSampler
from config import SEED, N_JOBS
from data_protocol import add_common_arguments, load_bundle
from training_common import fit_tabular

PARAM_GRID = {'n_estimators': [600, 800, 1000], 'max_depth': [3],
              'learning_rate': [.02]}
EXTENDED_GRID = {'n_estimators': [200, 400, 600, 800, 1000],
    'max_depth': [2, 3, 4, 5], 'learning_rate': [.01, .02, .05, .1],
    'subsample': [.7, 1.], 'colsample_bytree': [.6, .8, 1.],
    'min_child_weight': [1, 3, 5], 'reg_lambda': [1., 5., 10.]}


def make_estimator(params):
    from xgboost import XGBRegressor
    return XGBRegressor(booster='gbtree', objective='reg:squarederror',
        max_leaves=127, seed=SEED, n_jobs=N_JOBS, colsample_bytree=.6,
        subsample=.7, eval_metric='rmse').set_params(**params)


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--extended-search', action='store_true',
                        help='另运行预先固定的32候选搜索，命名XGBoost_extended，不替代主模型')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    bundle = load_bundle(args.results)
    searches = {'XGBoost': list(ParameterGrid(PARAM_GRID))}
    if args.extended_search:
        searches['XGBoost_extended'] = list(ParameterSampler(EXTENDED_GRID, n_iter=32, random_state=SEED))
    for name, candidates in searches.items():
        if args.check_only:
            print(f'{name}: {len(candidates)} candidates; no model fitted.')
        else:
            for target in args.targets:
                fit_tabular(bundle, target, name, make_estimator, candidates)


if __name__ == '__main__':
    main()
