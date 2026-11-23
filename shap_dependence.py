"""SHAP依赖散点图；不绘制不适用的普通bootstrap置信带。"""
import argparse
import pandas as pd
from config import FEATURES, DISPLAY, TARGET_LABELS
from data_protocol import add_common_arguments,load_bundle
from explanation_common import load_explanation,choose_features


def main():
    parser=add_common_arguments(argparse.ArgumentParser(description=__doc__),model=True)
    parser.add_argument('--top-n',type=int,choices=range(1,14),default=6)
    parser.add_argument('--features',nargs='+',choices=FEATURES)
    args=parser.parse_args();bundle=load_bundle(args.results)
    from plot_utils import plt,panel_grid,save_figure
    for target in args.targets:
        _,X,values,importance,out=load_explanation(bundle,args.model,target)
        selected=choose_features(importance,args.features,args.top_n)
        source=[]
        for start in range(0,len(selected),6):
            page=selected[start:start+6]
            fig,axes,rg,cg=panel_grid(len(page),f'{TARGET_LABELS[target]} | {args.model}','SHAP contribution')
            for ax,feature in zip(axes,page):
                j=FEATURES.index(feature)
                ax.axhline(0,color='#929292',lw=.6)
                ax.scatter(X[:,j],values[:,j],s=8,alpha=.65,color='#287FA3',edgecolors='none')
                ax.set_xlabel(DISPLAY.get(feature,feature),fontsize=11)
                if feature in ['Day of Week','Hour of Day']:
                    from matplotlib.ticker import MaxNLocator
                    ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
                for i,x,y in zip(bundle.protocol['train_ids'],X[:,j],values[:,j]):
                    source.append({'source_row':i,'feature':feature,'feature_value':x,'SHAP':y})
            save_figure(fig,out,f'SHAP_dependence_page{start//6+1}',axes,row_groups=rg,column_groups=cg)
        pd.DataFrame(source).to_csv(out/'SHAP_dependence_source.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':main()
