"""有实际意义的数据/时间隔离检查；不重新训练任何真实模型。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from config import DEFAULT_DATA, FEATURES, AUTHOR_CONFIRMED_CSV_SHA256
from data_protocol import prepare, sha256
from training_common import cross_validate,scores
from train_lstm import prepare_fold
from explanation_common import ranked_pairs,feature_grid,pdp_1d,pdp_2d


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='paper5_revised_test_')
        # Public tests use generated fixtures, never private observations or results.
        # An independent artificial calendar, unrelated to author observations.
        times=pd.date_range('2020-01-01', '2020-01-30 23:00:00', freq='h')
        times=times[times.day!=16]
        cls.synthetic_times=times
        n=len(times);index=np.arange(n)
        values={feature:1+(index%31)/10 for feature in FEATURES}
        values['Day of Week']=times.dayofweek+1
        values['Hour of Day']=times.hour
        values.update(date=times.strftime('%Y-%m-%d'),
                      SGE=.1+(index%17)/1000, EGE=.12+(index%19)/1000,
                      GiniStart=.45+(index%23)/1000, GiniEnd=.46+(index%29)/1000,
                      ordercount=100+index, **{'Travel Distance':500+index})
        cls.csv=Path(cls.temp.name)/'synthetic.csv'
        pd.DataFrame(values).to_csv(cls.csv,index=False)
        cls.bundle=prepare(cls.csv,Path(cls.temp.name)/'results')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_common_sample_and_folds(self):
        # 720 artificial hours minus one missing day = 696 source rows.
        # The first 23 rows and the first 23 rows after the gap are ineligible.
        expected_eligible=list(range(23,360))+list(range(383,696))
        expected_train=[i for i in expected_eligible if i<556]
        expected_test=list(range(556,696))
        protocol=self.bundle.protocol
        self.assertEqual(protocol['features'],FEATURES)
        self.assertEqual(protocol['timestamps'],[t.isoformat() for t in self.synthetic_times])
        self.assertEqual(protocol['eligible_ids'],expected_eligible)
        self.assertEqual(protocol['train_ids'],expected_train)
        self.assertEqual(protocol['test_ids'],expected_test)
        self.assertEqual(len(protocol['train_ids']),510)
        self.assertEqual(len(protocol['test_ids']),140)
        for fold_number,fold in enumerate(protocol['cv_folds'],1):
            end=85*fold_number
            self.assertEqual(fold['train_ids'],expected_train[:end])
            self.assertEqual(fold['validation_ids'],expected_train[end:end+85])
        self.assertEqual(self.bundle.X.shape,(696,13))
        self.assertTrue(set(['ordercount','Travel Distance','Trip Volume']).isdisjoint(self.bundle.frame.columns))

    def test_duplicate_hours_are_rejected(self):
        raw=pd.read_csv(self.csv)
        raw=pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
        path=Path(self.temp.name)/'duplicate.csv';raw.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'时间戳重复'):
            prepare(path,Path(self.temp.name)/'duplicate_results')

    def test_temporal_cv_does_not_fit_validation_or_test(self):
        # 记录接口调用的数据；不是训练真正的回归模型。
        calls=[]
        class RecordingEstimator:
            def fit(self,X,y):calls.append((X.copy(),y.copy()));return self
            def predict(self,X):return np.full(len(X),.2)
        def factory(params):return RecordingEstimator()
        cv,_=cross_validate(self.bundle,'EGE',factory,[{}])
        self.assertEqual(len(calls),5)
        for (X,y),fold in zip(calls,self.bundle.protocol['cv_folds']):
            np.testing.assert_array_equal(X,self.bundle.X[fold['train_ids']])
            np.testing.assert_array_equal(y,self.bundle.y('EGE')[fold['train_ids']])
            self.assertLess(max(fold['train_ids']),min(fold['validation_ids']))
        self.assertEqual(cv.n.tolist(),[85]*5)

    def test_lstm_preprocessing_uses_training_information_only(self):
        for f in self.bundle.protocol['cv_folds']:
            tr,va=f['train_ids'],f['validation_ids']
            Xt,Xv,yt,sx,sy,audit=prepare_fold(self.bundle,'EGE',tr,va)
            self.assertEqual(Xt.shape,(len(tr),24,13))
            self.assertEqual(Xv.shape,(85,24,13))
            self.assertLessEqual(max(audit['X_scaler_fit_rows']),max(tr))
            self.assertTrue(set(audit['X_scaler_fit_rows']).isdisjoint(va))
            self.assertEqual(audit['y_scaler_fit_rows'],tr)
            np.testing.assert_allclose(sy.data_min_,[self.bundle.y('EGE')[tr].min()])
            np.testing.assert_allclose(sy.data_max_,[self.bundle.y('EGE')[tr].max()])

    def test_missing_day_windows_are_not_bridged(self):
        p=self.bundle.protocol;times=pd.to_datetime(p['timestamps'])
        for i in p['eligible_ids']:
            self.assertTrue((times[i-23:i+1].to_series().diff().dropna()==pd.Timedelta(hours=1)).all())
        self.assertEqual(len(p['excluded_target_rows']),46)

    def test_metric_consistency_and_negative_r2(self):
        y=np.array([0.,1.,2.]);prediction=np.array([4.,5.,6.])
        m=scores(y,prediction)
        self.assertAlmostEqual(m['MAE'],4)
        self.assertAlmostEqual(m['RMSE'],4)
        self.assertLess(m['R2'],0)

    def test_rank_pair_rule_and_discrete_grids(self):
        importance=pd.DataFrame({'feature':FEATURES,'rank':range(1,14)})
        pairs=ranked_pairs(importance)
        self.assertEqual(pairs[:3],[(FEATURES[0],FEATURES[i]) for i in [1,2,3]])
        self.assertEqual(pairs[-1],(FEATURES[2],FEATURES[5]))
        self.assertEqual(len(set(pairs)),9)
        for feature in ['Day of Week','Hour of Day']:
            grid=feature_grid(self.bundle.X[self.bundle.protocol['train_ids']],feature)
            np.testing.assert_array_equal(grid,np.unique(self.bundle.X[:,FEATURES.index(feature)]))
            self.assertTrue((grid%1==0).all())

    def test_pdp_averages_all_background_and_keeps_input_unchanged(self):
        # 使用有闭式解的模型接口核验PDP语义，而不是比较另一份相同循环。
        class KnownResponse:
            def predict(self,X):return X[:,0]*2 + X[:,2]*3 + X[:,3]
        X=self.bundle.X[self.bundle.protocol['train_ids']].copy();original=X.copy()
        gx=np.array([1.,7.]);gy=np.array([30.,100.])
        z=pdp_2d(KnownResponse(),X,FEATURES[0],FEATURES[2],gx,gy)
        expected=2*gx[None,:]+3*gy[:,None]+X[:,3].mean()
        np.testing.assert_allclose(z,expected)
        line=pdp_1d(KnownResponse(),X,FEATURES[0],gx)
        np.testing.assert_allclose(line,2*gx+3*X[:,2].mean()+X[:,3].mean())
        np.testing.assert_array_equal(X,original)


class OriginalDataRegressionTests(unittest.TestCase):
    def test_author_data_aggregate_protocol(self):
        if not DEFAULT_DATA.is_file():
            self.skipTest('Author CSV not supplied; synthetic protocol tests still run.')
        if sha256(DEFAULT_DATA)!=AUTHOR_CONFIRMED_CSV_SHA256:
            self.skipTest('Local CSV differs from author data; do not apply its reference counts.')
        with tempfile.TemporaryDirectory(prefix='paper5_author_regression_') as directory:
            bundle=prepare(DEFAULT_DATA,Path(directory)/'results')
            self.assertEqual(bundle.X.shape,(336,13))
            self.assertEqual(len(bundle.protocol['train_ids']),222)
            self.assertEqual(len(bundle.protocol['test_ids']),68)
            self.assertEqual([len(f['train_ids']) for f in bundle.protocol['cv_folds']],
                             [37,74,111,148,185])
            self.assertEqual([len(f['validation_ids']) for f in bundle.protocol['cv_folds']],
                             [37]*5)


if __name__=='__main__':unittest.main()
