"""LSTM：13变量连续24小时输入，固定配置，五折评价及末段测试。"""
from __future__ import annotations
from runtime import configure
configure()
import argparse
import random
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import FEATURES, LSTM_SEED, SEQUENCE_LENGTH
from data_protocol import add_common_arguments, load_bundle, sha256, write_json
from training_common import scores, prediction_records

PARAMETERS = {'units': 128, 'dropout_rate': .3, 'learning_rate': .01,
              'batch_size': 32, 'epochs': 50, 'layers': 1,
              'optimizer': 'Adam', 'loss': 'MSE', 'lstm_activation': 'tanh'}


def make_model():
    import tensorflow as tf
    tf.keras.backend.clear_session()
    random.seed(LSTM_SEED)
    np.random.seed(LSTM_SEED)
    tf.keras.utils.set_random_seed(LSTM_SEED)
    inputs = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, len(FEATURES)))
    x = tf.keras.layers.LSTM(PARAMETERS['units'], activation='tanh')(inputs)
    x = tf.keras.layers.Dropout(PARAMETERS['dropout_rate'])(x)
    output = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=PARAMETERS['learning_rate']),
                  loss='mse', metrics=['mae'])
    return model


def prepare_fold(bundle, target, train_ids, evaluation_ids):
    X, y = bundle.X, bundle.y(target)
    assert max(train_ids) < min(evaluation_ids)
    # X缩放器仅拟合该折训练窗口真正涉及的历史输入，绝不拟合验证输入。
    fit_ids = np.unique(np.concatenate([np.arange(i-23, i+1) for i in train_ids]))
    assert fit_ids.max() <= max(train_ids)
    assert set(fit_ids).isdisjoint(evaluation_ids)
    scaler_X = MinMaxScaler().fit(X[fit_ids])
    scaler_y = MinMaxScaler().fit(y[np.array(train_ids), None])
    scaled_X = scaler_X.transform(X).astype(np.float32)
    scaled_y = scaler_y.transform(y[:, None]).astype(np.float32)
    def windows(ids):
        times = pd.to_datetime(bundle.frame.timestamp)
        for i in ids:
            assert (times.iloc[i-23:i+1].diff().dropna() == pd.Timedelta(hours=1)).all()
        return np.stack([scaled_X[i-23:i+1] for i in ids])
    audit = {'training_target_rows': list(train_ids), 'evaluation_target_rows': list(evaluation_ids),
        'X_scaler_fit_rows': fit_ids.tolist(), 'y_scaler_fit_rows': list(train_ids),
        'X_data_min': scaler_X.data_min_.tolist(), 'X_data_max': scaler_X.data_max_.tolist(),
        'y_data_min': scaler_y.data_min_.tolist(), 'y_data_max': scaler_y.data_max_.tolist(),
        'features': FEATURES, 'model_input_shape': [None, 24, 13],
        'early_stopping': False, 'evaluation_used_for_training': False}
    return windows(train_ids), windows(evaluation_ids), scaled_y[train_ids], scaler_X, scaler_y, audit


def fit_one(bundle, target, train_ids, evaluation_ids, name, final=False):
    import tensorflow as tf
    out = bundle.root/'training'/'LSTM'/target
    out.mkdir(parents=True, exist_ok=True)
    Xt, Xe, yt, sx, sy, audit = prepare_fold(bundle, target, train_ids, evaluation_ids)
    model = make_model()
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 2
    dataset = tf.data.Dataset.from_tensor_slices((Xt, yt))
    # 仅打乱训练序列的批次顺序；不打乱窗口内部顺序或时间划分。
    dataset = dataset.shuffle(len(Xt), seed=LSTM_SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(PARAMETERS['batch_size']).with_options(options)
    history = model.fit(dataset, epochs=PARAMETERS['epochs'], verbose=0)
    train_pred = sy.inverse_transform(model(Xt, training=False).numpy()).ravel().astype(float)
    eval_pred = sy.inverse_transform(model(Xe, training=False).numpy()).ravel().astype(float)
    pd.DataFrame({'epoch': range(1, PARAMETERS['epochs']+1), **history.history}).to_csv(out/f'{name}_history.csv', index=False)
    write_json(out/f'{name}_audit.json', audit)
    if final:
        path = bundle.root/'models'/f'{target}__LSTM.keras'
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        scaler_path = path.with_suffix('.scalers.joblib')
        joblib.dump({'X': sx, 'y': sy, 'features': FEATURES}, scaler_path)
        write_json(path.with_suffix('.json'), {'target': target, 'model': 'LSTM',
            'features': FEATURES, 'protocol_sha256': bundle.protocol_hash,
            'model_sha256': sha256(path), 'scalers_sha256': sha256(scaler_path),
            'parameters': PARAMETERS, 'seed': LSTM_SEED, 'candidate_count': 1,
            'hyperparameter_search': False, 'sequence_length': 24,
            'sequence_includes_current_X': True, 'training_rows': train_ids,
            'test_rows': evaluation_ids})
    return train_pred, eval_pred, audit


def main():
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    bundle = load_bundle(args.results)
    if args.check_only:
        for target in args.targets:
            prepare_fold(bundle, target, bundle.protocol['train_ids'], bundle.protocol['test_ids'])
        print('LSTM input/scalers: 24 x 13; five shared folds; one fixed configuration; no model fitted.')
        return
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    for target in args.targets:
        start, rows = time.monotonic(), []
        out = bundle.root/'training'/'LSTM'/target
        for f in bundle.protocol['cv_folds']:
            _, predicted, _ = fit_one(bundle, target, f['train_ids'], f['validation_ids'], f'fold{f["fold"]}')
            rows.append({'fold': f['fold'], 'candidate': 0, 'train_n': len(f['train_ids']),
                         **scores(bundle.y(target)[f['validation_ids']], predicted)})
            print(f'LSTM {target} fold {f["fold"]} complete.', flush=True)
        cv = pd.DataFrame(rows)
        cv.to_csv(out/'cross_validation.csv', index=False)
        mean_mse = float(cv.MSE.mean())
        trp, tep, _ = fit_one(bundle, target, bundle.protocol['train_ids'], bundle.protocol['test_ids'], 'final', final=True)
        predictions, metrics = prediction_records(bundle, target, 'LSTM', trp, tep)
        metrics['CV_RMSE'] = np.sqrt(mean_mse)
        predictions.to_csv(out/'predictions.csv', index=False)
        metrics.to_csv(out/'metrics.csv', index=False)
        metadata_path = bundle.root/'models'/f'{target}__LSTM.json'
        import json
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        metadata.update(mean_validation_mse=mean_mse, CV_RMSE=float(np.sqrt(mean_mse)),
                        elapsed_seconds=time.monotonic()-start)
        write_json(metadata_path, metadata)


if __name__ == '__main__':
    main()
