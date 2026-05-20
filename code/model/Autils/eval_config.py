"""统一评估与数据路径：各模型脚本从此处读取，避免分散修改。"""
from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径与数据（相对项目根 ballMill/）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
FILE_PATH = str(_PROJECT_ROOT / "data" / "processed_1min_data.csv")
OUTPUT_DIR = str(_PROJECT_ROOT / "output")

# ---------------------------------------------------------------------------
# 时序与划分
# ---------------------------------------------------------------------------
TARGET_COL = "Current_A"
LAG_STEPS = 60
# 预测未来N分钟（例如 10，在多步预测中代表预测未来1~10分钟）
PREDICT_HORIZON = 15
# ---------------------------------------------------------------------------
# 工业命中率与评估指标
# ---------------------------------------------------------------------------
INDUSTRIAL_HIT_REL_TOLERANCE = 0.005

PRED_STEPS = PREDICT_HORIZON

TRAIN_RATIO = 0.8
VAL_RATIO = 0.01
# 测试集占比 = 1 - TRAIN_RATIO - VAL_RATIO

PLOT_TAIL = 500
GLOBAL_SEED = 42


def eval_config_tag() -> str:
    """与 model_eval_compare_60_5_0.005.txt 一致的参数标签。"""
    return f"{LAG_STEPS}_{PREDICT_HORIZON}_{INDUSTRIAL_HIT_REL_TOLERANCE:g}"


def get_metrics_report_path() -> str:
    return str(_PROJECT_ROOT / "output" / f"model_eval_compare_{eval_config_tag()}.txt")


METRICS_REPORT_PATH = get_metrics_report_path()


def set_global_seed(seed: int | None = None) -> int:
    """固定 Python / NumPy / PyTorch 随机性，便于复现实验。"""
    seed = int(GLOBAL_SEED if seed is None else seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed


def make_dataloader_generator(seed: int | None = None):
    try:
        import torch

        g = torch.Generator()
        g.manual_seed(int(GLOBAL_SEED if seed is None else seed))
        return g
    except Exception:
        return None


def sequence_split_indices(
    n_raw: int,
    *,
    lag_steps: int | None = None,
    predict_horizon: int | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> tuple[int, int, int]:
    """原始序列长度 -> (n_windows, train_end, val_end)。"""
    lag = int(lag_steps if lag_steps is not None else LAG_STEPS)
    h = int(predict_horizon if predict_horizon is not None else PREDICT_HORIZON)
    tr = float(train_ratio if train_ratio is not None else TRAIN_RATIO)
    vr = float(val_ratio if val_ratio is not None else VAL_RATIO)
    n_windows = n_raw - lag - h + 1
    if n_windows <= 0:
        raise ValueError("序列过短，无法满足 LAG_STEPS 与 PREDICT_HORIZON")
    train_end = int(n_windows * tr)
    val_end = int(n_windows * (tr + vr))
    return n_windows, train_end, val_end


def raw_train_end_index(
    train_end: int,
    *,
    lag_steps: int | None = None,
    predict_horizon: int | None = None,
) -> int:
    """仅用训练窗口覆盖到的原始行（含标签）拟合 scaler，避免泄露。"""
    lag = int(lag_steps if lag_steps is not None else LAG_STEPS)
    h = int(predict_horizon if predict_horizon is not None else PREDICT_HORIZON)
    return train_end + lag + h - 1


def fit_minmax_scalers_train_only(
    df_values: np.ndarray,
    y_values: np.ndarray,
    *,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    lag_steps: int | None = None,
    predict_horizon: int | None = None,
):
    """在训练段拟合 MinMaxScaler，再变换全量（无验证/测试统计量泄露）。"""
    from sklearn.preprocessing import MinMaxScaler

    _, train_end, _ = sequence_split_indices(
        len(df_values),
        lag_steps=lag_steps,
        predict_horizon=predict_horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    raw_end = raw_train_end_index(
        train_end, lag_steps=lag_steps, predict_horizon=predict_horizon
    )
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(df_values[:raw_end])
    scaler_y.fit(y_values[:raw_end])
    return (
        scaler_x.transform(df_values),
        scaler_y.transform(y_values),
        scaler_x,
        scaler_y,
    )


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

def horizon_step_index(step: int | None = None) -> int:
    """多步矩阵列下标：1 表示第 1 分钟，-1 或 None 表示最后一步 PREDICT_HORIZON。"""
    if step is None or step == -1:
        return PREDICT_HORIZON - 1
    s = int(step)
    if not 1 <= s <= PREDICT_HORIZON:
        raise ValueError(f"step 须在 1..{PREDICT_HORIZON} 内")
    return s - 1


def target_time_index(win_idx: np.ndarray, step_idx: int, *, axis_mode: str = "sequence") -> np.ndarray:
    """
    sequence: 滑窗样本 i 的第 step 列 → 时刻 i + LAG_STEPS + step_idx
    lag_table: 表格 lag 行 g 的第 step 列 → 时刻 g + step_idx（特征已含 LAG）
    """
    if axis_mode == "sequence":
        return win_idx + LAG_STEPS + step_idx
    if axis_mode == "lag_table":
        return win_idx + step_idx
    raise ValueError(f"未知 axis_mode: {axis_mode}")


def extract_aligned_horizon_series(
    y_true_matrix,
    y_pred_matrix,
    *,
    step: int | None = None,
    plot_tail: int | None = None,
    window_offset: int = 0,
    axis_mode: str = "sequence",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将 (n_windows, H) 矩阵按「目标时刻」对齐为 1D 序列。"""
    yt_m = np.asarray(y_true_matrix, dtype=float)
    yp_m = np.asarray(y_pred_matrix, dtype=float)
    if yt_m.ndim == 1:
        yt_m = yt_m.reshape(-1, 1)
    if yp_m.ndim == 1:
        yp_m = yp_m.reshape(-1, 1)
    step_idx = horizon_step_index(step)
    tail = int(plot_tail if plot_tail is not None else PLOT_TAIL)
    n = yt_m.shape[0]
    start = max(0, n - tail)
    win_idx = np.arange(start, n) + int(window_offset)
    x = target_time_index(win_idx, step_idx, axis_mode=axis_mode)
    return x, yt_m[start:, step_idx], yp_m[start:, step_idx]


def prediction_plot_basename(model_slug: str, *, step: int | None = None) -> str:
    """例: gru_predict_60_5_0.005_h5.png"""
    step_no = PREDICT_HORIZON if step is None or step == -1 else int(step)
    return f"{model_slug}_predict_{eval_config_tag()}_h{step_no}.png"


def save_multistep_horizon_plot(
    y_true_matrix,
    y_pred_matrix,
    *,
    model_slug: str,
    model_label: str | None = None,
    step: int | None = None,
    plot_tail: int | None = None,
    window_offset: int = 0,
    axis_mode: str = "sequence",
    out_filename: str | None = None,
) -> str:
    """测试集末尾曲线：x 为目标时刻索引，y 为该时刻真值/预测值。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step_no = PREDICT_HORIZON if step is None or step == -1 else int(step)
    x, yt, yp = extract_aligned_horizon_series(
        y_true_matrix,
        y_pred_matrix,
        step=step,
        plot_tail=plot_tail,
        window_offset=window_offset,
        axis_mode=axis_mode,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = out_filename or prediction_plot_basename(model_slug, step=step)
    out_path = os.path.join(OUTPUT_DIR, fname)
    _hit_band_pct = INDUSTRIAL_HIT_REL_TOLERANCE * 100
    label = model_label or model_slug

    plt.figure(figsize=(15, 5))
    plt.plot(x, yt, label=f"True (step {step_no})", color="blue", alpha=0.6)
    plt.plot(x, yp, label=f"{label} Pred (step {step_no})", color="red", linestyle="--", alpha=0.8)
    upper = yt * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
    lower = yt * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
    plt.fill_between(x, lower, upper, color="gray", alpha=0.15, label=f"±{_hit_band_pct:g}% band")
    plt.xlabel("Target time index (1-min grid)")
    plt.title(
        f"{label} — lag={LAG_STEPS} horizon={PREDICT_HORIZON} "
        f"hit±{INDUSTRIAL_HIT_REL_TOLERANCE:g} — last {len(x)} targets (step {step_no})"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_prediction_plot(
    y_true,
    y_pred,
    *,
    out_filename: str,
    title: str,
    pred_label: str,
    plot_tail: int | None = None,
    x_values: np.ndarray | None = None,
) -> str:
    """兼容旧接口；若传入矩阵请优先用 save_multistep_horizon_plot。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tail = plot_tail if plot_tail is not None else PLOT_TAIL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_filename)

    yt = np.asarray(y_true, dtype=float).ravel()[-tail:]
    yp = np.asarray(y_pred, dtype=float).ravel()[-tail:]
    x = (
        np.asarray(x_values, dtype=float).ravel()[-tail:]
        if x_values is not None
        else np.arange(len(yt))
    )
    _hit_band_pct = INDUSTRIAL_HIT_REL_TOLERANCE * 100

    plt.figure(figsize=(15, 5))
    plt.plot(x, yt, label="True Current", color="blue", alpha=0.6)
    plt.plot(x, yp, label=pred_label, color="red", linestyle="--", alpha=0.8)
    upper = yt * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
    lower = yt * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
    plt.fill_between(x, lower, upper, color="gray", alpha=0.15, label=f"±{_hit_band_pct:g}% band")
    plt.title(title)
    plt.xlabel("Target time index (1-min grid)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def release_gpu_memory() -> None:
    """在当前进程内尽量归还 PyTorch 占用的 CUDA 缓存块（配合 ``gc.collect()``）。

    要点：

    - **进程正常退出**时，操作系统会回收该进程占用的整块 GPU 显存；子进程跑完脚本后通常不必依赖本函数。
    - ``torch.cuda.empty_cache()`` 主要把空闲块还给 **PyTorch 的显存池**，便于**同一进程**里连续跑多段训练时降低峰值碎片；不能替代进程退出，也**不能修复**驱动层的 GPU 失联。
    - ``nvidia-smi`` 出现 **Unable to determine the device handle ... Unknown Error** 多与驱动/GPU 通信异常、Xid、硬件或电源有关；应查 ``dmesg``、日志与驱动版本，不能单靠 ``empty_cache()`` 解决。
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def xgb_preferred_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"