"""训练期SHAP总结图：默认解释XGBoost，保存全部13列逐行贡献。"""
from runtime import configure
configure()
import argparse
import numpy as np
import pandas as pd
from config import FEATURES, DISPLAY, TARGET_LABELS
from data_protocol import add_common_arguments, load_bundle, sha256, write_json
from training_common import load_tabular_model
from explanation_common import explanation_folder


def compute_shap(estimator, X, model_name):
    import shap
    X = np.asarray(X, float)
    if model_name in ['RF', 'CatBoost']:
        # 路径依赖模式兼容这些已保存模型；仍独立对原生predict严格校验。
        explainer = shap.TreeExplainer(estimator, feature_perturbation='tree_path_dependent', model_output='raw')
        method, background_rows = 'tree_path_dependent', 0
    else:
        # 禁止默认背景抽样成100行：明确保留全部训练背景。
        masker = shap.maskers.Independent(X, max_samples=len(X))
        if model_name == 'OLS':
            explainer = shap.LinearExplainer(estimator, masker)
        else:
            explainer = shap.TreeExplainer(estimator, data=masker,
                feature_perturbation='interventional', model_output='raw')
        method, background_rows = 'interventional', len(X)
        assert np.asarray(explainer.masker.data if model_name=='OLS' else explainer.data).shape[0] == len(X)
    values = np.asarray(explainer.shap_values(X), float)
    baseline = np.asarray(explainer.expected_value, float).reshape(-1)
    if values.shape != X.shape or len(baseline) != 1:
        raise ValueError('仅支持单目标回归的13列SHAP输出。')
    predicted = np.asarray(estimator.predict(X)).ravel()
    error = float(np.max(np.abs(baseline[0] + values.sum(1) - predicted)))
    if not np.allclose(baseline[0] + values.sum(1), predicted, atol=2e-6, rtol=2e-5):
        raise ValueError(f'SHAP加和不等于模型预测：最大误差{error}')
    return values, float(baseline[0]), method, background_rows, error


def draw_summary(X, values, importance, out, target, model):
    from plot_utils import plt, save_figure, packed_offsets
    from matplotlib.patches import Patch
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    colors = {'Time features': '#DEE3E8', 'Weather conditions': '#CDE3ED', 'Air pollution': '#F7DECE'}
    def group(f):
        return 'Time features' if f in ['Day of Week', 'Hour of Day'] else ('Weather conditions' if f in ['Temperature','Wind Speed','Pressure','Humidity','Precipitation'] else 'Air pollution')
    order = importance.sort_values('rank').feature.tolist()
    positions = np.arange(13)
    fig = plt.figure(figsize=(183/25.4, 168/25.4))
    ax = fig.add_axes([.25,.17,.61,.64]); top = ax.twiny()
    top.barh(positions, importance.sort_values('rank').mean_abs_SHAP,
             height=.75, color=[colors[group(f)] for f in order], alpha=.8)
    top.set_xlim(0, max(float(importance.mean_abs_SHAP.max())*1.12, 1e-12))
    top.set_xlabel('Mean |SHAP value|', fontsize=11, color='#277C99', labelpad=8)
    top.tick_params(axis='x', colors='#277C99', labelsize=9.5)
    top.spines['top'].set_color('#277C99'); top.spines[['right','bottom','left']].set_visible(False)
    ax.set_zorder(top.get_zorder()+1); ax.patch.set_visible(False)
    cmap = plt.get_cmap('coolwarm')
    for row, feature in enumerate(order):
        j = FEATURES.index(feature)
        low, high = np.quantile(X[:,j], [.05,.95])
        color_values = np.clip((X[:,j]-low)/(high-low), 0, 1) if high>low else np.full(len(X), .5)
        ax.scatter(values[:,j], row+packed_offsets(values[:,j]), c=color_values,
                   cmap=cmap, vmin=0, vmax=1, s=8, alpha=.8, linewidths=0, zorder=3)
    ax.axvline(0, color='#808080', lw=.6, zorder=1)
    ax.set_yticks(positions, [DISPLAY.get(f,f) for f in order], fontsize=11)
    ax.set_ylim(12.7,-.7); top.set_ylim(ax.get_ylim())
    ax.set_xlabel('SHAP value (contribution)', fontsize=11, color='#B34843', labelpad=7)
    ax.tick_params(axis='x', colors='#B34843', labelsize=9.5)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.spines[['top','right']].set_visible(False)
    cax=fig.add_axes([.89,.25,.023,.45])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1), cmap=cmap), cax=cax)
    cb.set_ticks([0,1], labels=['Low','High']);cb.ax.tick_params(labelsize=9.5)
    fig.text(.902,.19,'Feature\nvalue',ha='center',va='center',fontsize=10.5)
    fig.text(.55,.96,f'{TARGET_LABELS[target]} | {model}',ha='center',fontsize=11)
    fig.legend(handles=[Patch(facecolor=v,label=k) for k,v in colors.items()],
               loc='center',bbox_to_anchor=(.54,.055),ncol=3,frameon=False,
               fontsize=9.5,handlelength=1.3,columnspacing=1.2)
    save_figure(fig,out,'SHAP_summary',[ax],exclude=[top,cax])


def main():
    parser=add_common_arguments(argparse.ArgumentParser(description=__doc__), model=True)
    parser.add_argument('--no-plots',action='store_true',help='只导出数值，不绘图')
    args=parser.parse_args();bundle=load_bundle(args.results)
    ids=bundle.protocol['train_ids'];X=bundle.X[ids]
    for target in args.targets:
        estimator,model_meta=load_tabular_model(bundle,target,args.model)
        out=explanation_folder(bundle,args.model,target);out.mkdir(parents=True,exist_ok=True)
        values,base,method,background,error=compute_shap(estimator,X,args.model)
        frame=pd.DataFrame(values,columns=FEATURES)
        frame.insert(0,'timestamp',bundle.frame.iloc[ids].timestamp.to_numpy())
        frame.insert(0,'source_row',ids)
        frame.to_csv(out/'shap_values.csv',index=False,encoding='utf-8-sig')
        bundle.frame.iloc[ids][['source_row','timestamp',*FEATURES]].to_csv(out/'explained_training_values.csv',index=False,encoding='utf-8-sig')
        importance=pd.DataFrame({'feature':FEATURES,'mean_abs_SHAP':np.abs(values).mean(0)})
        importance=importance.sort_values('mean_abs_SHAP',ascending=False,kind='stable').reset_index(drop=True)
        importance['rank']=np.arange(1,14)
        importance.to_csv(out/'shap_importance.csv',index=False,encoding='utf-8-sig')
        write_json(out/'shap_metadata.json',{'target':target,'model':args.model,'features':FEATURES,
            'protocol_sha256':bundle.protocol_hash,'model_sha256':model_meta['model_sha256'],
            'explained_rows':len(ids),'background_rows':background,'method':method,
            'expected_value':base,'max_additivity_error':error,
            'shap_values_sha256':sha256(out/'shap_values.csv'),
            'importance_sha256':sha256(out/'shap_importance.csv'),
            'ranking_scope':'Training period only; stable sort for exact ties.',
            'background_note':'Full training rows explicitly supplied for interventional/linear modes; RF/CatBoost path-dependent modes use training-tree cover counts.',
            'causal_effect':False,'confidence_intervals':False})
        if not args.no_plots:draw_summary(X,values,importance,out,target,args.model)
        print(f'SHAP {target}/{args.model}: all {len(ids)} training rows; maximum additivity error={error:.3g}',flush=True)


if __name__=='__main__':main()
