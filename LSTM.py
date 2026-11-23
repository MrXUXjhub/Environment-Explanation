"""LSTM: standalone 24-hour input model with training-only scaling."""

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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})

# 1. Edit the input path and target list here.
ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / 'data/GE.csv'
TARGETS = ['SGE', 'EGE', 'GiniStart', 'GiniEnd']
FEATURES = ['Day of Week', 'Hour of Day', 'O3', 'PM2.5', 'PM10', 'SO2',
            'NO2', 'CO', 'Temperature', 'Wind Speed', 'Pressure',
            'Humidity', 'Precipitation']
WINDOW = 24
N_SPLITS = 5
SAVE_PLOTS = True

def read_data():
    """Sort hours and select identical target times for all six models."""
    raw = pd.read_csv(DATA_FILE)
    data_hash = hashlib.sha256(Path(DATA_FILE).read_bytes()).hexdigest()
    hours = pd.to_numeric(raw['Hour of Day'], errors='raise')
    if not ((hours >= 0) & (hours <= 23) & (hours % 1 == 0)).all():
        raise ValueError('Hour of Day must contain integers 0--23.')
    if 'timestamp' in raw:
        times = pd.to_datetime(raw['timestamp'], errors='raise')
    else:
        times = pd.to_datetime(raw['date'], errors='raise').dt.normalize() + pd.to_timedelta(hours, unit='h')
    # Apply the author's confirmed year correction only to the exact source file.
    if data_hash == '7ef31080ed421e61560c292fe60866da8b61fec013bfa6235111c0dddf8cee0b' and set(times.dt.year) == {2023}:
        times = times.map(lambda t: t.replace(year=2017))
    if times.duplicated().any() or not np.array_equal(times.dt.hour, hours):
        raise ValueError('Duplicate timestamps or inconsistent hourly labels.')
    raw = raw.assign(timestamp=times).sort_values('timestamp', kind='stable').reset_index(drop=True)
    numeric = raw[FEATURES + TARGETS].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError('Predictors and targets must be finite; no full-data imputation is used.')
    X = numeric[FEATURES].to_numpy(float)
    times = raw['timestamp']
    eligible = np.array([i for i in range(WINDOW - 1, len(raw))
                         if times.iloc[i-WINDOW+1:i+1].diff().dropna().eq(pd.Timedelta(hours=1)).all()])
    cut = int(len(raw) * 0.8)
    train_ids, test_ids = eligible[eligible < cut], eligible[eligible >= cut]
    if len(train_ids) < N_SPLITS + 1 or not len(test_ids):
        raise ValueError('Not enough continuous hourly records for temporal validation.')
    return numeric, X, times, train_ids, test_ids, data_hash


def metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSE': float(np.sqrt(mse)), 'MSE': float(mse),
            'R2': float(r2_score(actual, predicted))}


def plot_predictions(predictions):
    """One figure per model, with shared axis labels and one legend."""
    cols = min(2, len(TARGETS)); rows = (len(TARGETS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 3.3 * rows + 0.4), squeeze=False)
    fig.subplots_adjust(left=.12, right=.98, bottom=.12, top=.88, wspace=.32, hspace=.38)
    used = []
    for ax, target in zip(axes.flat, TARGETS):
        used.append(ax)
        table = predictions[predictions.target == target]
        for split, color, label in [('train', '#287FA3', 'Train'), ('test', '#D34A4A', 'Test')]:
            subset = table[table.split == split]
            ax.scatter(subset.actual, subset.predicted, s=12, alpha=.7, color=color, label=label)
        low = min(table.actual.min(), table.predicted.min())
        high = max(table.actual.max(), table.predicted.max())
        pad = max((high-low)*.05, 1e-6)
        ax.plot([low-pad, high+pad], [low-pad, high+pad], 'k--', lw=1, label='y = x')
        ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad), title=target)
        ax.grid(alpha=.2)
    for ax in axes.flat[len(TARGETS):]:
        fig.delaxes(ax)
    handles, labels = used[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.55, .98), ncol=3, frameon=False)
    fig.supxlabel('Actual Values', fontsize=13)
    fig.supylabel('Predicted Values', fontsize=13)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(OUTPUT_DIR / f'predictions.{extension}', dpi=600)
    plt.close(fig)

# 2. LSTM settings. Scaling is fitted separately within each training fold.
MODEL_NAME = 'LSTM'
OUTPUT_DIR = ROOT / 'results/LSTM'
UNITS, DROPOUT, LEARNING_RATE = 128, 0.3, 0.01
EPOCHS, BATCH_SIZE, SEED = 50, 32, 5


def fit_lstm(X, y, train_ids, evaluate_ids):
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    fit_rows = np.unique(np.concatenate([np.arange(i-WINDOW+1, i+1) for i in train_ids]))
    scaler_X = MinMaxScaler().fit(X[fit_rows])
    scaler_y = MinMaxScaler().fit(y[train_ids, None])
    scaled_X = scaler_X.transform(X).astype(np.float32)
    scaled_y = scaler_y.transform(y[:, None]).astype(np.float32)
    train_X = np.stack([scaled_X[i-WINDOW+1:i+1] for i in train_ids])
    evaluate_X = np.stack([scaled_X[i-WINDOW+1:i+1] for i in evaluate_ids])
    inputs = tf.keras.layers.Input(shape=(WINDOW, len(FEATURES)))
    hidden = tf.keras.layers.LSTM(UNITS, activation='tanh')(inputs)
    hidden = tf.keras.layers.Dropout(DROPOUT)(hidden)
    outputs = tf.keras.layers.Dense(1)(hidden)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse')
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    dataset = tf.data.Dataset.from_tensor_slices((train_X, scaled_y[train_ids]))
    dataset = dataset.shuffle(len(train_ids), seed=SEED).batch(BATCH_SIZE).with_options(options)
    model.fit(dataset, epochs=EPOCHS, verbose=0, shuffle=False)
    train_prediction = scaler_y.inverse_transform(model(train_X, training=False).numpy()).ravel()
    evaluation_prediction = scaler_y.inverse_transform(model(evaluate_X, training=False).numpy()).ravel()
    return model, scaler_X, scaler_y, train_prediction, evaluation_prediction


def main():
    data, X, times, train_ids, test_ids, data_hash = read_data()
    print(f'LSTM: {len(train_ids)} train / {len(test_ids)} test targets; {WINDOW} x {len(FEATURES)} inputs.')
    if '--check-only' in sys.argv:
        return
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics, all_predictions = [], []
    for target in TARGETS:
        y = data[target].to_numpy(float)
        cv_scores = []
        # 3. Evaluate the same five expanding folds as the table models.
        for fold, (a, b) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS, gap=0).split(train_ids), 1):
            _, _, _, _, prediction = fit_lstm(X, y, train_ids[a], train_ids[b])
            cv_scores.append({'fold': fold, **metrics(y[train_ids[b]], prediction)})
            print(target, 'fold', fold, 'complete', flush=True)
        pd.DataFrame(cv_scores).to_csv(OUTPUT_DIR / f'{target}_cv.csv', index=False)
        cv_rmse = float(np.sqrt(np.mean([score['MSE'] for score in cv_scores])))
        # 4. Fixed epochs; the final test period is not used for early stopping.
        model, sx, sy, train_prediction, test_prediction = fit_lstm(X, y, train_ids, test_ids)
        model.save(OUTPUT_DIR / f'{target}_model.keras')
        joblib.dump({'scaler_X': sx, 'scaler_y': sy, 'features': FEATURES,
                     'target': target, 'window': WINDOW, 'data_sha256': data_hash},
                    OUTPUT_DIR / f'{target}_scalers.joblib')
        for split, ids, prediction in [('train', train_ids, train_prediction), ('test', test_ids, test_prediction)]:
            score = {'target': target, 'model': MODEL_NAME, 'split': split,
                     'n': len(ids), **metrics(y[ids], prediction), 'CV_RMSE': cv_rmse}
            all_metrics.append(score)
            all_predictions.append(pd.DataFrame({'target': target, 'split': split,
                'source_row': ids, 'timestamp': times.iloc[ids].to_numpy(),
                'actual': y[ids], 'predicted': prediction}))
            print(target, split, score)
    predictions = pd.concat(all_predictions, ignore_index=True)
    pd.DataFrame(all_metrics).to_csv(OUTPUT_DIR / 'metrics.csv', index=False)
    predictions.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)
    if SAVE_PLOTS:
        plot_predictions(predictions)


if __name__ == '__main__':
    main()
