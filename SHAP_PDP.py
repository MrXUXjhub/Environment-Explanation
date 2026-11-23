"""SHAP and PDP in one standalone file. Run a table-model script first."""
import os
import sys
from pathlib import Path

# CPU settings; on macOS, reuse scikit-learn's installed OpenMP library.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
if __name__ == '__main__' and sys.platform == 'darwin' and os.environ.get('GE_OPENMP_READY') != '1':
    lib = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/sklearn/.dylibs'
    if (lib / 'libomp.dylib').is_file():
        env = dict(os.environ, GE_OPENMP_READY='1')
        env['DYLD_LIBRARY_PATH'] = str(lib) + (':' + env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache/matplotlib'))

import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})


from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

# 1. Edit the model and targets here. No model fitting occurs in this file.
ROOT = Path(__file__).resolve().parent
MODEL_NAME = 'XGBoost'  # Also supports OLS, RF, CatBoost and LightGBM.
MODEL_DIR = ROOT / 'results' / MODEL_NAME
OUTPUT_DIR = ROOT / 'results/SHAP_PDP' / MODEL_NAME
TARGETS = ['SGE', 'EGE', 'GiniStart', 'GiniEnd']
FEATURES = ['Day of Week', 'Hour of Day', 'O3', 'PM2.5', 'PM10', 'SO2',
            'NO2', 'CO', 'Temperature', 'Wind Speed', 'Pressure',
            'Humidity', 'Precipitation']
TOP_N = 6
GRID_POINTS = 31
SAVE_PLOTS = True
DISPLAY = {'Day of Week': 'Day of week', 'Hour of Day': 'Hour of day',
           'Wind Speed': 'Wind speed', 'O3': 'O₃', 'NO2': 'NO₂',
           'SO2': 'SO₂', 'PM2.5': 'PM₂.₅', 'PM10': 'PM₁₀'}
TARGET_LABELS = {'SGE': 'Origin GE', 'EGE': 'Destination GE',
                 'GiniStart': 'Origin Gini', 'GiniEnd': 'Destination Gini'}
plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 11,
                     'axes.titlesize': 11, 'xtick.labelsize': 9.5,
                     'ytick.labelsize': 9.5, 'axes.linewidth': .6})


def save_figure(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'{name}.{extension}', dpi=600)
    plt.close(fig)


def panel_grid(title, ylabel, count):
    rows = (count + 2) // 3
    height = 25 + 62 * rows
    fig = plt.figure(figsize=(183/25.4, height/25.4))
    axes = []
    for k in range(count):
        row, col = divmod(k, 3)
        ax = fig.add_axes([(20+58*col)/183, (22+62*(rows-row-1))/height,
                           40/183, 40/height])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.spines[['top', 'right']].set_visible(False)
        ax.annotate(chr(97+k), (0, 1), xycoords='axes fraction', xytext=(0, 6),
                    textcoords='offset points', fontsize=11, fontweight='bold', va='bottom')
        axes.append(ax)
    fig.text(.5, (height-6)/height, title, ha='center', va='top', fontsize=11)
    fig.text(4/183, .5, ylabel, ha='center', va='center', rotation=90, fontsize=11)
    return fig, axes


def feature_grid(X, feature):
    values = X[:, FEATURES.index(feature)]
    unique = np.unique(values)
    if feature in ['Day of Week', 'Hour of Day'] or len(unique) <= GRID_POINTS:
        return unique
    return np.linspace(*np.quantile(values, [.05, .95]), GRID_POINTS)


def partial_dependence(model, X, feature_x, grid_x, feature_y=None, grid_y=None):
    """Replace input(s) in every training-background row, predict, then average."""
    if feature_y is None:
        output = []
        for x in grid_x:
            modified = X.copy()
            modified[:, FEATURES.index(feature_x)] = x
            output.append(float(np.asarray(model.predict(modified)).mean()))
        return np.asarray(output)
    output = np.empty((len(grid_y), len(grid_x)))
    for row, y in enumerate(grid_y):
        for col, x in enumerate(grid_x):
            modified = X.copy()
            modified[:, FEATURES.index(feature_x)] = x
            modified[:, FEATURES.index(feature_y)] = y
            output[row, col] = float(np.asarray(model.predict(modified)).mean())
    return output

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

def packed_offsets(values, bins=60):
    """仅调整点的纵向排布；不改变任何SHAP横坐标或删除观测。"""
    values = np.asarray(values)
    if np.ptp(values) == 0:
        labels = np.zeros(len(values), dtype=int)
    else:
        labels = np.minimum(((values-values.min())/np.ptp(values)*bins).astype(int), bins-1)
    offsets = np.zeros(len(values))
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        if len(ids) > 1:
            lanes = np.array([0 if j==0 else ((j+1)//2)*(1 if j%2 else -1) for j in range(len(ids))])
            offsets[ids] = lanes/max(abs(lanes).max(), 1)*.28
    return offsets

def draw_summary(X, values, importance, out, target, model):
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
    save_figure(fig, out, 'SHAP_summary')

def draw_surfaces(surfaces,out,target,model,contours=False):
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
    save_figure(fig, out, 'PDP_2D_3x3')
    return [norm.vmin,norm.vmax]


def main():
    if MODEL_NAME not in ['OLS', 'RF', 'CatBoost', 'LightGBM', 'XGBoost']:
        raise ValueError('Choose a table regression model for SHAP/PDP.')
    if not 1 <= TOP_N <= 13 or GRID_POINTS < 2:
        raise ValueError('TOP_N must be 1--13 and GRID_POINTS must be at least 2.')
    for target in TARGETS:
        # 2. Load the fitted model and the exact training background it saved.
        path = MODEL_DIR / f'{target}_model.joblib'
        if not path.is_file():
            raise FileNotFoundError(f'Run {MODEL_NAME}.py first; missing {path}')
        package = joblib.load(path)
        if package['features'] != FEATURES or package['target'] != target or package['model_name'] != MODEL_NAME:
            raise ValueError('Model, target or input-column order does not match.')
        model, X = package['model'], np.asarray(package['X_train'], float)
        if X.ndim != 2 or X.shape[1] != 13 or not np.isfinite(X).all():
            raise ValueError('The model background must have 13 finite predictor columns.')
        out = OUTPUT_DIR / target
        out.mkdir(parents=True, exist_ok=True)

        # 3. SHAP summary and dependence plots; all training rows are retained.
        values, baseline, method, background_n, error = compute_shap(model, X, MODEL_NAME)
        pd.DataFrame(values, columns=FEATURES).to_csv(out / 'SHAP_values.csv', index=False)
        pd.DataFrame(X, columns=FEATURES).to_csv(out / 'training_background.csv', index=False)
        importance = pd.DataFrame({'feature': FEATURES, 'mean_abs_SHAP': np.abs(values).mean(0)})
        importance = importance.sort_values('mean_abs_SHAP', ascending=False, kind='stable').reset_index(drop=True)
        importance['rank'] = np.arange(1, 14)
        importance.to_csv(out / 'SHAP_importance.csv', index=False)
        order = importance.feature.tolist()
        if SAVE_PLOTS:
            draw_summary(X, values, importance, out, target, MODEL_NAME)
        pdp_1d_rows = []
        for start in range(0, TOP_N, 6):
            features = order[start:min(start+6, TOP_N)]
            if SAVE_PLOTS:
                fig_s, axes_s = panel_grid(f'{TARGET_LABELS[target]} | {MODEL_NAME}', 'SHAP contribution', len(features))
                fig_p, axes_p = panel_grid(f'{TARGET_LABELS[target]} | {MODEL_NAME}', f'Predicted {TARGET_LABELS[target]}', len(features))
            for k, feature in enumerate(features):
                j = FEATURES.index(feature)
                grid = feature_grid(X, feature)
                predictions = partial_dependence(model, X, feature, grid)
                pdp_1d_rows.extend({'feature': feature, 'value': float(v), 'prediction': float(p)}
                                   for v, p in zip(grid, predictions))
                if SAVE_PLOTS:
                    axes_s[k].axhline(0, color='#929292', lw=.6)
                    axes_s[k].scatter(X[:, j], values[:, j], s=8, alpha=.65, color='#287FA3', edgecolors='none')
                    axes_p[k].plot(grid, predictions, 'o' if feature == 'Day of Week' else '-', color='#287FA3', lw=1.2, markersize=3)
                    for ax in [axes_s[k], axes_p[k]]:
                        ax.set_xlabel(DISPLAY.get(feature, feature), fontsize=11)
                        if feature in ['Day of Week', 'Hour of Day']:
                            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            if SAVE_PLOTS:
                save_figure(fig_s, out, f'SHAP_dependence_{start//6+1}')
                save_figure(fig_p, out, f'PDP_1D_{start//6+1}')
        pd.DataFrame(pdp_1d_rows).to_csv(out / 'PDP_1D.csv', index=False)

        # 4. Three leading features, each paired with the next three ranked features.
        pairs = [(order[r], order[r+c+1]) for r in range(3) for c in range(3)]
        surfaces, pdp_2d_rows = [], []
        for panel, (feature_x, feature_y) in enumerate(pairs):
            gx, gy = feature_grid(X, feature_x), feature_grid(X, feature_y)
            z = partial_dependence(model, X, feature_x, gx, feature_y, gy)
            surfaces.append({'pair': (feature_x, feature_y), 'x': gx, 'y': gy, 'z': z})
            pdp_2d_rows.extend({'panel': chr(97+panel), 'feature_x': feature_x, 'feature_y': feature_y,
                'value_x': float(x), 'value_y': float(y), 'prediction': float(z[row, col])}
                for row, y in enumerate(gy) for col, x in enumerate(gx))
            print(target, 'PDP pair', panel+1, '/ 9 complete', flush=True)
        pd.DataFrame(pdp_2d_rows).to_csv(out / 'PDP_2D.csv', index=False)
        limits = [min(float(s['z'].min()) for s in surfaces), max(float(s['z'].max()) for s in surfaces)]
        if SAVE_PLOTS:
            limits = draw_surfaces(surfaces, out, target, MODEL_NAME)
        (out / 'settings.json').write_text(json.dumps({'model': MODEL_NAME, 'target': target,
            'features': FEATURES, 'explained_rows': len(X), 'expected_value': baseline,
            'shap_method': method, 'explicit_shap_background_rows': background_n,
            'max_additivity_error': error, 'PDP_background_rows': len(X), 'pairs': pairs,
            'PDP_color_limits': limits, 'confidence_intervals': False,
            'data_sha256': package['data_sha256']}, indent=2), encoding='utf-8')
        print(target, 'SHAP/PDP complete; no model was refitted.', flush=True)


if __name__ == '__main__':
    main()
