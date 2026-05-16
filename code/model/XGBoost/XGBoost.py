import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from Aprocess import noiseReduce as NR
from Autils.eval_config import (
    FILE_PATH,
    LAG_STEPS,
    METRICS_REPORT_PATH,
    OUTPUT_DIR,
    PLOT_TAIL,
    PREDICT_HORIZON,
    TARGET_COL,
    TRAIN_RATIO,
    VAL_RATIO,
    INDUSTRIAL_HIT_REL_TOLERANCE,
    industrial_hit_rate_pct,
    append_model_eval_report,
    create_lag_features_df_multistep,
    ensure_output_dir,
    print_standard_eval_report,
    regression_metrics,
    release_gpu_memory,
    xgb_preferred_device,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 强制启用 GPU（你有 2 张卡，自动用第 0 张，速度最快）
# ==========================================
_xgb_device = "cuda"  # 强制 GPU 模式！

# ==========================================
# 2. 数据加载与预处理
# ==========================================
print("正在加载全量数据...")
df = pd.read_csv(FILE_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df.set_index("time", inplace=True)
df = df.ffill().bfill()

print("正在进行工业数据去噪...")
df = NR.clean_industrial_data(df)

# ==========================================
# 3. 特征工程（连续多步目标）
# ==========================================
print(f"构造特征：过去 {LAG_STEPS} 分钟 → 未来连续 {PREDICT_HORIZON} 分钟...")
df_processed, target_cols = create_lag_features_df_multistep(
    df, TARGET_COL, LAG_STEPS, PREDICT_HORIZON
)

# ==========================================
# 4. 数据集划分
# ==========================================
total_samples = len(df_processed)
train_size = int(total_samples * TRAIN_RATIO)
val_size = int(total_samples * VAL_RATIO)

train_data = df_processed.iloc[:train_size]
val_data = df_processed.iloc[train_size : train_size + val_size]
test_data = df_processed.iloc[train_size + val_size :]

X_train, y_train = train_data.drop(columns=target_cols), train_data[target_cols]
X_val, y_val = val_data.drop(columns=target_cols), val_data[target_cols]
X_test, y_test = test_data.drop(columns=target_cols), test_data[target_cols]

# ==========================================
# 5. ✅ XGBoost GPU 加速版（原生多目标）
# ==========================================
print(f"\n🔥 开始 GPU 训练 XGBoost 多目标模型 (一次性预测 {PREDICT_HORIZON} 步)...")

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,

    # ====================
    # ✅ GPU 核心配置
    # ====================
    tree_method="hist",      # 必须
    device="cuda",           # 启用 GPU
    multi_strategy="multi_output_tree",
    early_stopping_rounds=50,
)

# 训练
model.fit(
    X_train, y_train, 
    eval_set=[(X_val, y_val)], 
    verbose=50
)

print(f"✅ XGBoost GPU 模型训练完成！")

# ==========================================
# 6. 评估（不变）
# ==========================================
print("\n在测试集上进行全序列预测与评估...")
y_pred_matrix = model.predict(X_test)
y_true_matrix = y_test.values

metrics = regression_metrics(y_true_matrix.flatten(), y_pred_matrix.flatten())
print_standard_eval_report(f"XGBoost-Native（连续 {PREDICT_HORIZON} 步）", metrics)
append_model_eval_report(
    "XGBoost-Native-MultiOutput",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

print("\n" + "=" * 40)
print(f" 各预测步长（1~{PREDICT_HORIZON} 分钟）独立指标分析")
print("=" * 40)
for step in range(PREDICT_HORIZON):
    yt = y_true_matrix[:, step]
    yp = y_pred_matrix[:, step]
    hit = industrial_hit_rate_pct(yt, yp)
    mae = mean_absolute_error(yt, yp)
    print(f"未来第 {step + 1:2d} 分钟 | 命中率: {hit:6.2f}% | MAE: {mae:.4f}")
print("=" * 40 + "\n")

# ==========================================
# 7. 可视化（不变）
# ==========================================
ensure_output_dir()

fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(
    model,
    max_num_features=20,
    importance_type="gain",
    title=f"XGBoost Feature Importance (Native Multi-output, gain)",
    ax=ax,
    color="coral",
)
imp_path = f"{OUTPUT_DIR}/xgboost_importance_native_multi.png"
plt.tight_layout()
plt.savefig(imp_path, dpi=150)
plt.close()
print(f"特征重要性图: {imp_path}")

y_true_last = y_true_matrix[:, -1][-PLOT_TAIL:]
y_pred_last = y_pred_matrix[:, -1][-PLOT_TAIL:]

plt.figure(figsize=(15, 5))
plt.plot(y_true_last, label=f"True (step {PREDICT_HORIZON})", color="blue", alpha=0.6)
plt.plot(y_pred_last, label=f"XGBoost Native Pred (step {PREDICT_HORIZON})", color="red", linestyle="--", alpha=0.8)
upper = y_true_last * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
lower = y_true_last * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
plt.fill_between(
    np.arange(len(y_true_last)),
    lower,
    upper,
    color="gray",
    alpha=0.15,
    label=f"±{INDUSTRIAL_HIT_REL_TOLERANCE * 100:g}% band",
)
plt.title(f"Ball Mill Current — XGBoost Native (H={PREDICT_HORIZON}, last {PLOT_TAIL})")
plt.legend()
plt.grid(True)
plt.tight_layout()
pred_png = f"{OUTPUT_DIR}/xgboost_predict_native_multi.png"
plt.savefig(pred_png, dpi=150)
plt.close()

print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")

try:
    del model
except NameError:
    pass
release_gpu_memory()
