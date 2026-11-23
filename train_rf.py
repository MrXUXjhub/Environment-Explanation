"""Random Forest：训练期五折时间验证选择参数。"""
from training_common import model_cli
from config import SEED, N_JOBS

PARAM_GRID = {'n_estimators': [200, 400], 'max_depth': [10, 15, 20],
              'min_samples_split': [2], 'min_samples_leaf': [1],
              'max_features': ['sqrt']}


def make_estimator(params):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=SEED, n_jobs=N_JOBS).set_params(**params)


if __name__ == '__main__':
    model_cli('RF', make_estimator, PARAM_GRID)
