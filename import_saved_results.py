"""导入此前已完成的13变量结果，供解释代码复用；不重新训练。"""
import argparse
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from config import FEATURES, MAIN_MODELS, TABULAR_MODELS
from data_protocol import add_common_arguments,load_bundle,sha256,write_json
from training_common import save_tabular_model,scores


def main():
    parser=add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--source-results',type=Path,required=True,help='此前paper5_no_mobility目录')
    parser.add_argument('--models',nargs='+',choices=MAIN_MODELS+['XGBoost_extended'],default=MAIN_MODELS)
    args=parser.parse_args();bundle=load_bundle(args.results);source=args.source_results.resolve()
    old=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    for key in ['features','train_ids','test_ids','timestamps']:
        if old[key]!=bundle.protocol[key]:raise ValueError(f'旧结果协议不匹配：{key}')
    if old.get('input_sha256')!=bundle.protocol['raw_csv_sha256']:
        raise ValueError('旧结果的原始CSV指纹不匹配。')
    for a,b in zip(old['cv_folds'],bundle.protocol['cv_folds']):
        for key in ['train_ids','validation_ids']:
            if a[key]!=b[key]:raise ValueError('旧结果五折索引不匹配。')
    old_data=pd.read_csv(source/'prepared_model_data.csv')
    if not np.allclose(old_data[FEATURES+bundle.protocol['targets']],bundle.frame[FEATURES+bundle.protocol['targets']],rtol=0,atol=1e-12):
        raise ValueError('旧建模数值与本次准备的数据不一致。')
    selected=json.loads((source/'temporal/selected_parameters.json').read_text())
    tabular_predictions=pd.read_csv(source/'temporal/predictions.csv')
    tabular_cv=pd.read_csv(source/'temporal/cross_validation.csv')
    for model in args.models:
        for target in args.targets:
            out=bundle.root/'training'/model/target;out.mkdir(parents=True,exist_ok=True)
            if model=='LSTM':
                audit=json.loads((source/'lstm'/f'{target}_final_audit.json').read_text())
                if audit['features']!=FEATURES or audit['model_input_shape']!=[None,24,13]:
                    raise ValueError('旧LSTM输入不是24×13。')
                scalers=joblib.load(source/'lstm'/f'{target}_scalers.joblib')
                fit_rows=np.unique(np.concatenate([np.arange(i-23,i+1) for i in bundle.protocol['train_ids']]))
                assert scalers['features']==FEATURES
                assert np.allclose(scalers['X'].data_min_,bundle.X[fit_rows].min(0))
                assert np.allclose(scalers['X'].data_max_,bundle.X[fit_rows].max(0))
                models=bundle.root/'models';models.mkdir(exist_ok=True)
                model_path=models/f'{target}__LSTM.keras'
                scaler_path=model_path.with_suffix('.scalers.joblib')
                shutil.copy2(source/'lstm'/f'{target}_model.keras',model_path)
                shutil.copy2(source/'lstm'/f'{target}_scalers.joblib',scaler_path)
                prediction=pd.read_csv(source/'lstm/predictions.csv')
                prediction=prediction[prediction.target==target]
                cv=pd.read_csv(source/'lstm/cv_scores.csv')
                cv=cv[cv.target==target].copy();cv['MSE']=cv.RMSE**2
                cv_mse=float(cv.MSE.mean())
                metadata={'target':target,'model':model,'features':FEATURES,
                    'protocol_sha256':bundle.protocol_hash,'model_sha256':sha256(model_path),
                    'scalers_sha256':sha256(scaler_path),'mean_validation_mse':cv_mse,
                    'CV_RMSE':float(np.sqrt(cv_mse)),'sequence_length':24,
                    'sequence_includes_current_X':True,'candidate_count':1,
                    'imported_previously_fitted_model':True,'models_fitted_during_import':0}
                write_json(model_path.with_suffix('.json'),metadata)
            else:
                choice=selected[f'{target}__{model}']
                model_source=source/'temporal'/f'{target}_{model}.joblib'
                estimator=joblib.load(model_source)
                cv=tabular_cv[(tabular_cv.target==target)&(tabular_cv.model==model)].copy()
                prediction=tabular_predictions[(tabular_predictions.target==target)&(tabular_predictions.model==model)].copy()
                cv_mse=choice['mean_validation_mse']
                metadata=save_tabular_model(bundle,target,model,estimator,choice['params'],cv_mse,
                    selected_candidate=choice['candidate'],imported_previously_fitted_model=True,
                    original_model_sha256=sha256(model_source),models_fitted_during_import=0)
                for split in ['train','test']:
                    d=prediction[prediction.split==split].sort_values('source_row')
                    ids=bundle.protocol['train_ids' if split=='train' else 'test_ids']
                    if d.source_row.tolist()!=ids:raise ValueError('旧预测目标索引不一致。')
                    check=np.asarray(estimator.predict(bundle.X[ids])).ravel()
                    if not np.allclose(check,d.predicted,rtol=0,atol=1e-10):
                        raise ValueError('导入模型无法复现原来保存的预测。')
            core=['target','model','split','source_row','timestamp','observed','predicted']
            prediction[core].to_csv(out/'predictions.csv',index=False)
            metric_rows=[]
            for split in ['train','test']:
                d=prediction[prediction.split==split].sort_values('source_row')
                ids=bundle.protocol['train_ids' if split=='train' else 'test_ids']
                if d.source_row.tolist()!=ids or not np.allclose(d.observed,bundle.y(target)[ids],rtol=0,atol=1e-12):
                    raise ValueError('旧预测目标或实际值不匹配。')
                metric_rows.append({'target':target,'model':model,'split':split,
                    **scores(d.observed,d.predicted),'CV_RMSE':float(np.sqrt(cv_mse))})
            pd.DataFrame(metric_rows).to_csv(out/'metrics.csv',index=False)
            cv.to_csv(out/'cross_validation.csv',index=False)
            print(f'Imported {model}/{target}; no model fitted.',flush=True)
    write_json(bundle.root/'import_audit.json',{'source_protocol_sha256':sha256(source/'protocol.json'),
        'matched_features_timestamps_and_folds':True,'models':args.models,
        'targets':args.targets,'models_fitted':0})


if __name__=='__main__':main()
