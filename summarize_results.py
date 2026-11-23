"""汇总真实逐行预测及精度，不指定任何模型必须排名第一。"""
import argparse
import numpy as np
import pandas as pd
from config import MAIN_MODELS,TARGETS
from data_protocol import add_common_arguments,load_bundle,write_json
from training_common import scores


def main():
    parser=add_common_arguments(argparse.ArgumentParser(description=__doc__))
    args=parser.parse_args();bundle=load_bundle(args.results)
    frames=[];metrics=[]
    for path in sorted((bundle.root/'training').glob('*/*/predictions.csv')):
        d=pd.read_csv(path)
        if d.empty or d.target.iloc[0] not in args.targets:continue
        target,model=d.target.iloc[0],d.model.iloc[0]
        score_path=path.parent/'metrics.csv'
        saved=pd.read_csv(score_path) if score_path.is_file() else None
        for split in ['train','test']:
            frame=d[d.split==split].sort_values('source_row')
            ids=bundle.protocol['train_ids' if split=='train' else 'test_ids']
            if frame.source_row.tolist()!=ids or frame.timestamp.tolist()!=[bundle.protocol['timestamps'][i] for i in ids]:
                raise ValueError(f'{model}/{target}/{split}使用了不同的目标记录。')
            if not np.allclose(frame.observed,bundle.y(target)[ids],rtol=0,atol=1e-12):
                raise ValueError('保存的实际值与数据不一致。')
            result={'target':target,'model':model,'split':split,**scores(frame.observed,frame.predicted)}
            if saved is not None:
                row=saved[saved.split==split].iloc[0]
                for key in ['MAE','RMSE','R2']:
                    if abs(float(row[key])-result[key])>1e-10:raise ValueError(f'保存精度不一致: {path}')
                if 'CV_RMSE' in row:result['CV_RMSE']=float(row.CV_RMSE)
            metrics.append(result)
        frames.append(d)
    if not frames:raise FileNotFoundError('没有可汇总的已保存预测。')
    out=bundle.root/'summary';out.mkdir(exist_ok=True)
    all_metrics=pd.DataFrame(metrics);pred=pd.concat(frames,ignore_index=True)
    all_metrics.to_csv(out/'all_metrics.csv',index=False,encoding='utf-8-sig')
    pred.to_csv(out/'all_predictions.csv',index=False,encoding='utf-8-sig')
    main_metrics=all_metrics[all_metrics.model.isin(MAIN_MODELS)]
    main_metrics.to_csv(out/'six_model_metrics.csv',index=False,encoding='utf-8-sig')
    all_metrics[all_metrics.split=='test'].to_csv(out/'test_metrics.csv',index=False,encoding='utf-8-sig')
    write_json(out/'verification.json',{'status':'PASS','metric_groups':len(all_metrics),
        'prediction_rows':len(pred),'all_models_share_target_timestamps':True,
        'metrics_recomputed_from_saved_predictions':True,'ranking_forced':False,
        'negative_R2_retained':True})
    print(main_metrics[main_metrics.split=='test'].to_string(index=False))


if __name__=='__main__':main()
