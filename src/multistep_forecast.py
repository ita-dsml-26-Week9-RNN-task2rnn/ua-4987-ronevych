from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def make_windows(series: np.ndarray, window: int, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(list(series[i:i+window]))
        y.append(list(series[i+window:i+window+horizon]))
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X[..., None], y


def time_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    n = len(X)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_model(
    window: int,
    output_dim: int,
    n_units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(n_units, input_shape=(window, 1)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


def train_model(
    series: np.ndarray,
    window: int,
    horizon: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    epochs: int = 25,
    batch_size: int = 64,
    seed: int = 42,
    verbose: int = 0,
) -> Tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    tf.keras.utils.set_random_seed(seed)
    
    X, y = make_windows(series, window, horizon=horizon)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_split(X, y, train_frac, val_frac)
    
    model = build_model(window, output_dim=horizon)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )
    
    return model, X_test, y_test


def recursive_rollout_one_step(
    model: tf.keras.Model,
    init_window: np.ndarray,
    horizon: int = 100,
) -> np.ndarray:
    window = init_window.copy().astype(np.float32)
    preds = []
    for _ in range(horizon):
        yhat = model.predict(window[None, ..., None], verbose=0)[0, 0]
        preds.append(yhat)
        window = np.concatenate([window[1:], [yhat]])
    return np.array(preds, dtype=np.float32)


def recursive_rollout_k_step_stride_k(
    model: tf.keras.Model,
    init_window: np.ndarray,
    k: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    window = init_window.copy().astype(np.float32)
    preds = []
    for _ in range(horizon // k):
        block = model.predict(window[None, ..., None], verbose=0)[0]
        preds.extend(block)
        window = np.concatenate([window[k:], block])
    return np.array(preds, dtype=np.float32)


def recursive_rollout_k_step_stride_1(
    model: tf.keras.Model,
    init_window: np.ndarray,
    k: int = 20,
    horizon: int = 100,
) -> np.ndarray:
    window = init_window.copy().astype(np.float32)
    preds = []
    for _ in range(horizon):
        block = model.predict(window[None, ..., None], verbose=0)[0]
        preds.append(block[0])
        window = np.concatenate([window[1:], [block[0]]])
    return np.array(preds, dtype=np.float32)


def horizon_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)}


def plot_rollouts(y_true: np.ndarray, preds: Dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(y_true, label="true", linewidth=2)
    for name, y_hat in preds.items():
        plt.plot(y_hat, label=name, alpha=0.9)
    plt.grid(True)
    plt.legend()
    plt.title("Multi-step rollout comparison")
    plt.show()


def visualize_forecasts(predictions_dict: Dict[str, np.ndarray], true_data: np.ndarray) -> None:
    horizon = len(true_data)
    x_axis = np.arange(horizon)
    
    preds_trimmed = {}
    for name, pred in predictions_dict.items():
        preds_trimmed[name] = pred[:horizon]
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(x_axis, true_data, color='black', linewidth=2.5, linestyle='-', 
             label='Ground Truth', zorder=5)
    
    strategy_colors = {
        'one_step': 'red',
        'k_step': 'blue',
        'rollout': 'green',
    }
    
    strategy_labels = {
        'one_step': 'One-step (Recursive, Stride=1)',
        'k_step': 'K-step (Multi-output, Stride=20)',
        'rollout': 'K-step Rollout (Stride=1)',
    }
    
    for strategy_name, pred_data in preds_trimmed.items():
        color = strategy_colors.get(strategy_name, 'gray')
        label = strategy_labels.get(strategy_name, strategy_name)
        plt.plot(x_axis, pred_data, color=color, linewidth=1.8, alpha=0.8, label=label)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlabel('Time Steps', fontsize=12, fontweight='bold')
    plt.ylabel('Signal Value', fontsize=12, fontweight='bold')
    plt.title('Forecast Drift Comparison (100-point Horizon)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("FORECAST DRIFT ANALYSIS - PERFORMANCE METRICS")
    print("="*60)
    
    for strategy_name, pred_data in preds_trimmed.items():
        label = strategy_labels.get(strategy_name, strategy_name)
        metrics = horizon_errors(true_data, pred_data)
        mae_val = metrics['mae']
        rmse_val = metrics['rmse']
        
        print(f"\n{label}:")
        print(f"  MAE:  {mae_val:.6f}")
        print(f"  RMSE: {rmse_val:.6f}")
    
    print("\n" + "="*60)


def _make_series(n: int = 2500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float32)
    x = 0.0009 * t + 2.0 * np.sin(2 * np.pi * t / 50.0) + 0.8 * np.sin(2 * np.pi * t / 16.0)
    x += rng.normal(0, 0.2, size=n).astype(np.float32)
    return x.astype(np.float32)


def demo() -> None:
    tf.keras.utils.set_random_seed(123)

    series = _make_series(n=2600, seed=123)
    window = 40
    k = 20
    H = 100

    one_model, X_test_1, y_test_1 = train_model(series, window=window, horizon=1, epochs=15, seed=123, verbose=0)
    k_model, X_test_k, y_test_k = train_model(series, window=window, horizon=k, epochs=15, seed=123, verbose=0)

    init_window = series[-(window + H) : -H]
    y_true = series[-H:]

    pred_1 = recursive_rollout_one_step(one_model, init_window, horizon=H)
    pred_k20 = recursive_rollout_k_step_stride_k(k_model, init_window, k=k, horizon=H)
    pred_k1 = recursive_rollout_k_step_stride_1(k_model, init_window, k=k, horizon=H)

    preds = {
        "one_step": pred_1,
        "k_step": pred_k20,
        "rollout": pred_k1,
    }

    visualize_forecasts(preds, y_true)


if __name__ == "__main__":
    demo()
