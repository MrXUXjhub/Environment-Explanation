"""分阶段运行；本脚本只有显式执行时才训练模型。"""
import argparse
import subprocess
import sys
from config import ROOT,DEFAULT_DATA,DEFAULT_RESULTS,TARGETS,TABULAR_MODELS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','prepare','check','fit','summary','explain','plots'],default='all')
    parser.add_argument('--data',type=__import__('pathlib').Path,default=DEFAULT_DATA)
    parser.add_argument('--results',type=__import__('pathlib').Path,default=DEFAULT_RESULTS)
    parser.add_argument('--year',type=int)
    parser.add_argument('--targets',nargs='+',choices=TARGETS,default=TARGETS)
    parser.add_argument('--explain-model',choices=TABULAR_MODELS,default='XGBoost')
    parser.add_argument('--grid-resolution',type=int,default=31)
    args=parser.parse_args()
    def run(script,*options):
        command=[sys.executable,str(ROOT/script),*map(str,options)]
        print('Running:',script,flush=True)
        subprocess.run(command,check=True)
    common=['--results',args.results,'--targets',*args.targets]
    if args.stage in ['all','prepare'] or (args.stage=='check' and not (args.results/'protocol.json').exists()):
        prepare=['--data',args.data,'--results',args.results]
        if args.year is not None:prepare+=['--year',args.year]
        run('prepare_data.py',*prepare)
    if args.stage in ['all','fit','check']:
        for script in ['train_ols.py','train_rf.py','train_catboost.py','train_lightgbm.py','train_xgboost.py','train_lstm.py']:
            run(script,*common,*(['--check-only'] if args.stage=='check' else []))
    if args.stage in ['all','summary']:run('summarize_results.py',*common)
    if args.stage in ['all','explain']:
        explain=[*common,'--model',args.explain_model]
        run('shap_analysis.py',*explain)
        run('shap_dependence.py',*explain)
        run('pdp_1d.py',*explain,'--grid-resolution',args.grid_resolution)
        run('pdp_2d.py',*explain,'--grid-resolution',args.grid_resolution)
    # 4×6图需要四个目标都已存在；部分目标训练时不自动调用。
    if args.stage=='plots' or (args.stage=='all' and set(args.targets)==set(TARGETS)):
        run('plot_predictions.py','--results',args.results)


if __name__=='__main__':main()
