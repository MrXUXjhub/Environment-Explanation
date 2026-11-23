"""真正的一维PDP：逐格替换输入，在全部训练背景上平均预测。"""
import argparse
import numpy as np
import pandas as pd
from config import FEATURES, DISPLAY, TARGET_LABELS
from data_protocol import add_common_arguments, load_bundle, write_json
from explanation_common import load_explanation, choose_features, feature_grid, pdp_1d


def main():
    parser=add_common_arguments(argparse.ArgumentParser(description=__doc__),model=True)
    parser.add_argument('--top-n',type=int,choices=range(1,14),default=6)
    parser.add_argument('--features',nargs='+',choices=FEATURES)
    parser.add_argument('--grid-resolution',type=int,default=31)
    args=parser.parse_args();bundle=load_bundle(args.results)
    from plot_utils import panel_grid,save_figure
    for target in args.targets:
        estimator,X,_,importance,out=load_explanation(bundle,args.model,target)
        selected=choose_features(importance,args.features,args.top_n)
        records=[]
        for start in range(0,len(selected),6):
            page=selected[start:start+6]
            fig,axes,rg,cg=panel_grid(len(page),f'{TARGET_LABELS[target]} | {args.model}',
                                     f'Predicted {TARGET_LABELS[target]}')
            for ax,feature in zip(axes,page):
                grid=feature_grid(X,feature,args.grid_resolution)
                z=pdp_1d(estimator,X,feature,grid)
                if feature=='Day of Week':
                    ax.plot(grid,z,'o',color='#287FA3',markersize=3)
                    ax.set_xticks(grid)
                else:
                    ax.plot(grid,z,color='#287FA3',lw=1.2)
                if feature=='Hour of Day':
                    from matplotlib.ticker import MaxNLocator
                    ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
                ax.set_xlabel(DISPLAY.get(feature,feature),fontsize=11)
                for value,predicted in zip(grid,z):
                    records.append({'feature':feature,'feature_value':float(value),
                        'average_prediction':float(predicted),'background_n':len(X)})
            save_figure(fig,out,f'PDP_1D_page{start//6+1}',axes,row_groups=rg,column_groups=cg)
        pd.DataFrame(records).to_csv(out/'PDP_1D_source.csv',index=False,encoding='utf-8-sig')
        write_json(out/'PDP_1D_metadata.json',{'features':selected,'background':'All training target rows',
            'background_n':len(X),'continuous_grid':'Training 5th—95th percentiles; unique observed values if fewer than resolution',
            'discrete_time_grid':'Observed integer weekday/hour codes; no fractional weekday interpolation',
            'grid_resolution':args.grid_resolution,'confidence_interval':False,
            'method':'For every grid value, replace that feature in every training-background row and average model predictions.',
            'is_SHAP_dependence':False,'causal_effect':False})


if __name__=='__main__':main()
