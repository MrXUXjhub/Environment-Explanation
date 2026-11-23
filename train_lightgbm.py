"""LightGBM：训练期五折时间验证选择参数。"""
from training_common import model_cli
from config import SEED, N_JOBS

PARAM_GRID = {'num_leaves': [31], 'learning_rate': [.01, .05, .1],
              'n_estimators': [100, 500], 'feature_fraction': [.8, 1.]}


def make_estimator(params):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=SEED, n_jobs=N_JOBS,
                         verbosity=-1).set_params(**params)


if __name__ == '__main__':
    model_cli('LightGBM', make_estimator, PARAM_GRID)
