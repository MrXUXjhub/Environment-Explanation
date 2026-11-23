"""CatBoost：全部输入为数值，训练期五折时间验证选择参数。"""
from training_common import model_cli
from config import SEED, N_JOBS

PARAM_GRID = {'iterations': [500, 1000], 'learning_rate': [.05],
              'depth': [4], 'l2_leaf_reg': [3, 5]}


def make_estimator(params):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(random_seed=SEED, thread_count=N_JOBS,
                             verbose=False, allow_writing_files=False).set_params(**params)


if __name__ == '__main__':
    model_cli('CatBoost', make_estimator, PARAM_GRID)
