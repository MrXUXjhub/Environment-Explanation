"""运行输入、共同划分、时间隔离、LSTM缩放及PDP语义检查，不训练真实模型。"""
from runtime import configure
configure()
import importlib.metadata
import sys
import unittest
from config import ROOT


def main():
    for name in ['numpy','pandas','scikit-learn','xgboost','lightgbm','catboost','shap','matplotlib','tensorflow']:
        try:version=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:version='not installed'
        print(f'{name}: {version}',flush=True)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
