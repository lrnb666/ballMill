"""统一评估与数据路径：各模型脚本从此处读取，避免分散修改。"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径与数据（相对项目根 ballMill/）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILE_PATH = str(_PROJECT_ROOT / "data" / "processed_1min_data.csv")
OUTPUT_DIR = str(_PROJECT_ROOT / "output")
METRICS_REPORT_PATH = str(_PROJECT_ROOT / "output" / "model_eval_compare_60_5_0.01.txt")

# ---------------------------------------------------------------------------
# 时序与划分
# ---------------------------------------------------------------------------
TARGET_COL = "Current_A"
LAG_STEPS = 60
# 预测未来N分钟（例如 10，在多步预测中代表预测未来1~10分钟）
PREDICT_HORIZON = 5
PRED_STEPS = PREDICT_HORIZON

TRAIN_RATIO = 0.8
VAL_RATIO = 0.01
# 测试集占比 = 1 - TRAIN_RATIO - VAL_RATIO

PLOT_TAIL = 500

# ---------------------------------------------------------------------------
# 工业命中率与评估指标
# ---------------------------------------------------------------------------
INDUSTRIAL_HIT_REL_TOLERANCE = 0.01

def industrial_hit_rate_pct(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    band = np.abs(y_true * INDUSTRIAL_HIT_REL_TOLERANCE)
    return float(np.mean(np.abs(y_true - y_pred) <= band) * 100.0)

def calculate_smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return float(np.mean(diff) * 100.0)

# 【新增】支持多步连续预测的数据构造函数
def create_sequences_multistep(X, y, time_steps: int, predict_horizon: int):
    """
    构造连续多步预测数据集: [samples, time_steps, feat] -> [samples, predict_horizon]
    目标为未来连续的 predict_horizon 步。
    """
    X = np.asarray(X)
    y = np.asarray(y)
    h = int(predict_horizon)
    if h < 1:
        raise ValueError("predict_horizon 必须 >= 1")
    n = len(X) - time_steps - h + 1
    if n <= 0:
        raise ValueError("序列长度过短：无法满足 LAG_STEPS 与 PREDICT_HORIZON")
    Xs, ys = [], []
    for i in range(n):
        Xs.append(X[i : i + time_steps])
        # 提取未来 h 步连续的标签，并展平为一维数组 (h,)
        ys.append(y[i + time_steps : i + time_steps + h].ravel())
    return np.array(Xs), np.array(ys)

def create_sequences(X, y, time_steps: int, predict_horizon: int):
    """(原有的单点预测) 目标为未来第 predict_horizon 步（从窗口末尾算起）"""
    X = np.asarray(X)
    y = np.asarray(y)
    h = int(predict_horizon)
    n = len(X) - time_steps - h + 1
    Xs, ys = [], []
    for i in range(n):
        Xs.append(X[i : i + time_steps])
        ys.append(y[i + time_steps + h - 1])
    return np.array(Xs), np.array(ys)

def create_lag_features_df(data, target_col: str, lag_steps: int, predict_horizon: int):
    import pandas as pd
    h = int(predict_horizon)
    df_features = pd.DataFrame(index=data.index)
    for col in data.columns:
        for i in range(1, lag_steps + 1):
            df_features[f"{col}_lag_{i}"] = data[col].shift(i)
    df_features["target"] = data[target_col].shift(-h)
    return df_features.dropna()


def create_lag_features_df_multistep(data, target_col: str, lag_steps: int, predict_horizon: int):
    """表格模型：过去 lag_steps 步特征 → 未来连续 1..predict_horizon 步目标。"""
    import pandas as pd
    h = int(predict_horizon)
    df_features = pd.DataFrame(index=data.index)
    for col in data.columns:
        for i in range(1, lag_steps + 1):
            df_features[f"{col}_lag_{i}"] = data[col].shift(i)
    target_cols = []
    for i in range(1, h + 1):
        t_col = f"target_step_{i}"
        df_features[t_col] = data[target_col].shift(-i)
        target_cols.append(t_col)
    return df_features.dropna(), target_cols

def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    smape = calculate_smape(y_true, y_pred)
    hit = industrial_hit_rate_pct(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape, "industrial_hit_pct": hit}

def print_standard_eval_report(model_label: str, metrics: dict) -> None:
    _hit_band_pct = INDUSTRIAL_HIT_REL_TOLERANCE * 100
    print("\n" + "=" * 40)
    print(f" 🚀 测试集最终评估报告 — {model_label}")
    print("=" * 40)
    print(f"[1] R²    (决定系数): {metrics['r2']:.4f}")
    print(f"[2] MAE   (平均绝对误差): {metrics['mae']:.4f}")
    print(f"[3] RMSE  (均方根误差): {metrics['rmse']:.4f}")
    print(f"[4] MSE   (均方误差): {metrics['mse']:.4f}")
    print(f"[5] SMAPE (对称绝对百分比误差): {metrics['smape']:.2f}%")
    print(f"[6] ±{_hit_band_pct:g}% 误差带工业命中率: {metrics['industrial_hit_pct']:.2f}%")
    print("=" * 40 + "\n")

def append_model_eval_report(model_name: str, metrics: dict, *, lag_steps: int, predict_horizon: int, data_path: str | None = None) -> None:
    os.makedirs(os.path.dirname(METRICS_REPORT_PATH) or ".", exist_ok=True)
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    path = data_path or FILE_PATH
    lines = [
        "", "=" * 72, f"[{model_name}]  {ts}",
        f"FILE_PATH={path}", f"LAG_STEPS={lag_steps}  PREDICT_HORIZON={predict_horizon}",
        f"INDUSTRIAL_HIT_REL_TOLERANCE={INDUSTRIAL_HIT_REL_TOLERANCE}", "-" * 72,
        f"R2={metrics['r2']:.6f}", f"MAE={metrics['mae']:.6f}", f"RMSE={metrics['rmse']:.6f}",
        f"MSE={metrics['mse']:.6f}", f"SMAPE={metrics['smape']:.4f}",
        f"IndustrialHit%={metrics['industrial_hit_pct']:.4f}", "=" * 72,
    ]
    with open(METRICS_REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def save_prediction_plot(y_true, y_pred, *, out_filename: str, title: str, pred_label: str, plot_tail: int | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tail = plot_tail if plot_tail is not None else PLOT_TAIL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_filename)

    yt = np.asarray(y_true, dtype=float).ravel()[-tail:]
    yp = np.asarray(y_pred, dtype=float).ravel()[-tail:]
    _hit_band_pct = INDUSTRIAL_HIT_REL_TOLERANCE * 100

    plt.figure(figsize=(15, 5))
    plt.plot(yt, label="True Current", color="blue", alpha=0.6)
    plt.plot(yp, label=pred_label, color="red", linestyle="--", alpha=0.8)
    upper = yt * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
    lower = yt * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
    plt.fill_between(np.arange(len(yt)), lower, upper, color="gray", alpha=0.15, label=f"±{_hit_band_pct:g}% band")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def xgb_preferred_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"