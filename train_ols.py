"""OLS：固定线性回归，在共同五折中评价并在完整训练期拟合。"""
from training_common import model_cli

PARAM_GRID = {}


def make_estimator(params):
    from sklearn.linear_model import LinearRegression
    return LinearRegression().set_params(**params)


if __name__ == '__main__':
    model_cli('OLS', make_estimator, PARAM_GRID)
