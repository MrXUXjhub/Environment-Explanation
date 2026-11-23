"""二维PDP：训练SHAP前3变量各与后续3变量配对，共9图及真实共用色标。"""
import argparse
import numpy as np
import pandas as pd
from config import DISPLAY, TARGET_LABELS
from data_protocol import add_common_arguments, load_bundle, write_json
from explanation_common import load_explanation, ranked_pairs, feature_grid, pdp_2d


def draw_surfaces(surfaces,out,target,model,contours=False):
    from plot_utils import plt,save_figure
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator
    low=min(float(s['z'].min()) for s in surfaces)
    high=max(float(s['z'].max()) for s in surfaces)
    if high==low:
        delta=max(abs(low)*.001,1e-6)
        low,high=low-delta,high+delta
    # 所有9个面板使用同一组真实预测值范围，不使用虚设的-1到1。
    norm=Normalize(vmin=low,vmax=high)
    W,H=183,220
    fig=plt.figure(figsize=(183/25.4,220/25.4))
    axes=[]
    for k,surface in enumerate(surfaces):
        r,c=divmod(k,3)
        ax=fig.add_axes([(15+58*c)/W,[156,94,32][r]/H,42/W,44.8/H])
        ax.set_gid(chr(97+k));axes.append(ax)
        gx,gy,z=surface['x'],surface['y'],surface['z']
        if len(gx)<2 or len(gy)<2:
            raise ValueError('排序前列存在训练期常量变量，无法形成二维曲面；请核查数据。')
        ax.pcolormesh(gx,gy,z,cmap='viridis',norm=norm,shading='nearest',rasterized=True)
        categorical='Day of Week' in surface['pair']
        if contours and not categorical and np.ptp(z)>0:
            levels=np.linspace(norm.vmin,norm.vmax,7)[1:-1]
            ax.contour(gx,gy,z,levels=levels,colors='#4D5D63',linewidths=.4)
        # 星期按类别格展示，不用连续等高线在星期编码间插值。
        ax.set_xlim((gx[0]-.5,gx[-1]+.5) if surface['pair'][0]=='Day of Week' else (gx[0],gx[-1]))
        ax.set_ylim((gy[0]-.5,gy[-1]+.5) if surface['pair'][1]=='Day of Week' else (gy[0],gy[-1]))
        ax.xaxis.set_major_locator(MaxNLocator(4,integer=surface['pair'][0] in ['Day of Week','Hour of Day']))
        ax.yaxis.set_major_locator(MaxNLocator(4,integer=surface['pair'][1] in ['Day of Week','Hour of Day']))
        if surface['pair'][0]=='Day of Week':ax.set_xticks(gx)
        if surface['pair'][1]=='Day of Week':ax.set_yticks(gy)
        ax.tick_params(labelsize=9.5,pad=2.5)
        ax.set_ylabel(DISPLAY.get(surface['pair'][1],surface['pair'][1]),fontsize=11,labelpad=3)
        ax.annotate(chr(97+k),(0,1),xycoords='axes fraction',xytext=(0,5),
                    textcoords='offset points',fontsize=11,fontweight='bold',va='bottom')
    for r in range(3):
        feature=surfaces[r*3]['pair'][0]
        fig.text(94/W,([156,94,32][r]-10.5)/H,DISPLAY.get(feature,feature),
                 ha='center',va='center',fontsize=11.5)
    fig.text(.5,214/H,f'{TARGET_LABELS[target]} | {model}',ha='center',va='center',fontsize=11)
    cax=fig.add_axes([65/W,7/H,70/W,3.2/H])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),cax=cax,orientation='horizontal')
    cb.locator=MaxNLocator(4);cb.update_ticks();cb.ax.tick_params(labelsize=9,pad=2)
    fig.text(100/W,14/H,f'Predicted {TARGET_LABELS[target]}',ha='center',va='center',fontsize=10.5)
    save_figure(fig,out,'PDP_2D_3x3',axes,exclude=[cax],
        row_groups=[{'panels':[chr(97+r*3+c) for c in range(3)]} for r in range(3)],
        column_groups=[{'panels':[chr(97+r*3+c) for r in range(3)]} for c in range(3)])
    return [norm.vmin,norm.vmax]


def main():
    parser=add_common_arguments(argparse.ArgumentParser(description=__doc__),model=True)
    parser.add_argument('--grid-resolution',type=int,default=31)
    parser.add_argument('--contours',action='store_true',help='可选叠加无文字等高线；星期面板不插值')
    args=parser.parse_args();bundle=load_bundle(args.results)
    for target in args.targets:
        estimator,X,_,importance,out=load_explanation(bundle,args.model,target)
        pairs=ranked_pairs(importance);surfaces=[];records=[]
        for k,(fx,fy) in enumerate(pairs):
            gx,gy=feature_grid(X,fx,args.grid_resolution),feature_grid(X,fy,args.grid_resolution)
            z=pdp_2d(estimator,X,fx,fy,gx,gy)
            surfaces.append({'pair':(fx,fy),'x':gx,'y':gy,'z':z})
            for row,y in enumerate(gy):
                for col,x in enumerate(gx):
                    records.append({'panel':chr(97+k),'feature_x':fx,'feature_y':fy,
                        'value_x':float(x),'value_y':float(y),
                        'average_prediction':float(z[row,col]),'background_n':len(X)})
            print(f'PDP {target}/{args.model}: pair {k+1}/9 {fx} x {fy}',flush=True)
        pd.DataFrame(records).to_csv(out/'PDP_2D_source.csv',index=False,encoding='utf-8-sig')
        color_limits=draw_surfaces(surfaces,out,target,args.model,args.contours)
        write_json(out/'PDP_2D_metadata.json',{'target':target,'model':args.model,
            'pairs':[list(p) for p in pairs],'selection':'Training-only mean absolute SHAP rank; first 3 anchors, next 3 after each anchor.',
            'background_n':len(X),'background':'All training target rows',
            'grid_resolution':args.grid_resolution,'common_true_prediction_color_limits':color_limits,
            'method':'Replace both features in every training-background row at each grid point, predict and average.',
            'weekday_continuously_interpolated':False,'uncertainty_estimated':False,
            'interaction_significance_tested':False,'causal_effect':False,
            'correlated_predictors_note':'Partial dependence can evaluate sparsely supported feature combinations.'})


if __name__=='__main__':main()
