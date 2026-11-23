"""六模型4×6拟合图：全部保存预测、共享坐标名称和单一图例。"""
import argparse
import numpy as np
import pandas as pd
from config import MAIN_MODELS,TARGETS,DEFAULT_RESULTS
from data_protocol import load_bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=__import__('pathlib').Path,default=DEFAULT_RESULTS)
    args=parser.parse_args();bundle=load_bundle(args.results)
    frames=[]
    for target in TARGETS:
        for model in MAIN_MODELS:
            path=bundle.root/'training'/model/target/'predictions.csv'
            if not path.is_file():raise FileNotFoundError(f'需先生成六种模型的四个目标预测：{path}')
            frame=pd.read_csv(path)
            for split in ['train','test']:
                ids=bundle.protocol['train_ids' if split=='train' else 'test_ids']
                assert frame[frame.split==split].source_row.tolist()==ids
            frames.append(frame)
    data=pd.concat(frames,ignore_index=True)
    from plot_utils import plt,save_figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    W,H=183,158;bottoms=[110,80,50,20];axes=[]
    fig=plt.figure(figsize=(183/25.4,158/25.4))
    for r,target in enumerate(TARGETS):
        family=TARGETS[:2] if r<2 else TARGETS[2:]
        values=data[data.target.isin(family)][['observed','predicted']].to_numpy()
        low,high=values.min(),values.max();span=max(high-low,1e-6);low-=span*.07;high+=span*.07
        ticks=np.linspace(low+(high-low)*.15,high-(high-low)*.15,3)
        for c,model in enumerate(MAIN_MODELS):
            ax=fig.add_axes([(33+25*c)/W,bottoms[r]/H,22/W,22/H]);axes.append(ax)
            ax.set_gid(f'r{r}c{c}')
            ax.plot([low,high],[low,high],'--',color='#333333',lw=.6)
            for split,color in [('train','#348DB5'),('test','#DC4650')]:
                d=data[(data.target==target)&(data.model==model)&(data.split==split)]
                ax.scatter(d.observed,d.predicted,s=2.4,alpha=.7,color=color,linewidths=0)
            ax.set_xlim(low,high);ax.set_ylim(low,high)
            ax.set_xticks(ticks,[f'{x:.2f}' for x in ticks]);ax.set_yticks(ticks,[f'{x:.2f}' for x in ticks])
            ax.tick_params(axis='x',labelbottom=r in [1,3],labelsize=9,pad=2)
            ax.tick_params(axis='y',labelleft=c==0,labelsize=9,pad=2)
            ax.grid(color='#D7D7D7',ls=':',lw=.3);ax.set_axisbelow(True)
        fig.text(20.6/W,(bottoms[r]+11)/H,'Origin' if r%2==0 else 'Destination',
                 ha='center',va='center',rotation=90,rotation_mode='anchor',fontsize=10.5)
    for label,y in [('Greenspace exposure',80),('Gini index',20)]:
        fig.add_artist(Rectangle((8.4/W,y/H),6.3/W,52/H,transform=fig.transFigure,facecolor='#EEEEEE',edgecolor='none'))
        fig.text(11.55/W,(y+26)/H,label,rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=11)
    fig.text(3.7/W,76/H,'Predicted Values',rotation=90,rotation_mode='anchor',ha='center',va='center',fontsize=13)
    fig.text(106.5/W,6.8/H,'Actual Values',ha='center',va='center',fontsize=13)
    for c,model in enumerate(MAIN_MODELS):
        fig.text((44+25*c)/W,138.5/H,model,ha='center',va='center',fontsize=11)
    fig.legend(handles=[Line2D([],[],color='#348DB5',marker='o',ls='none',label='Train'),
        Line2D([],[],color='#DC4650',marker='o',ls='none',label='Test'),
        Line2D([],[],color='#333333',ls='--',label='y = x')],loc='center',
        bbox_to_anchor=(106.5/W,151/H),ncol=3,frameon=False,fontsize=11)
    out=bundle.root/'figures'
    data.to_csv(bundle.root/'prediction_figure_source.csv',index=False,encoding='utf-8-sig')
    save_figure(fig,out,'six_model_predictions',axes,
        row_groups=[{'panels':[f'r{r}c{c}' for c in range(6)]} for r in range(4)],
        column_groups=[{'panels':[f'r{r}c{c}' for r in range(4)]} for c in range(6)])


if __name__=='__main__':main()
