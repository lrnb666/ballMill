import warnings
import os

import xgboost as xgb  # 替换为 xgboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from Aprocess import noiseReduce as NR

# 引入必要的评估与配置
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
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 数据加载与预处理
# ==========================================
print("正在加载数据...")
df = pd.read_csv(FILE_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df.set_index("time", inplace=True)
df = df.ffill().bfill()
print(f"数据加载完成，共 {len(df)} 条记录。")

print("正在进行工业数据去噪...")
df = NR.clean_industrial_data(df)

# ==========================================
# 2. 特征工程 (使用多步构造)
# ==========================================
print(f"正在构造特征：过去 {LAG_STEPS} 分钟 → 未来连续 {PREDICT_HORIZON} 分钟...")
df_processed, target_cols = create_lag_features_df_multistep(df, TARGET_COL, LAG_STEPS, PREDICT_HORIZON)
print(f"特征构造完成，特征维度：{df_processed.shape[1] - len(target_cols)} 列，目标维度：{len(target_cols)} 列。")

# ==========================================
# 3. 数据集划分
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

print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# ==========================================
# 4. XGBoost 训练 (原生多目标回归策略)
# ==========================================
print(f"\n开始训练 XGBoost 原生多目标预测模型 (Horizon={PREDICT_HORIZON})...")

# 定义模型
# multi_strategy="multi_output_tree" 是核心，它让一棵树同时预测多个目标
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",              # 多目标模式必须使用 hist
    multi_strategy="multi_output_tree", # 关键参数：原生多输出
    n_jobs=-1,
)

# 训练模型（直接喂入整个 y_train 矩阵）
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    early_stopping_rounds=50,
    verbose=50
)

print(f"✅ 模型训练完成，最佳迭代次数: {model.get_booster().best_iteration}")

# ==========================================
# 5. 评估
# ==========================================
print("\n在测试集上进行全序列预测与评估...")

# XGBoost 直接返回 (N, PREDICT_HORIZON) 的矩阵
y_pred_matrix = model.predict(X_test)
y_true_matrix = y_test.values

# 全局平均指标计算
metrics = regression_metrics(y_true_matrix.flatten(), y_pred_matrix.flatten())
print_standard_eval_report(f"XGBoost-Native (连续 {PREDICT_HORIZON} 步整体平均)", metrics)
append_model_eval_report(
    "XGBoost-MultiOutput", metrics, 
    lag_steps=LAG_STEPS, predict_horizon=PREDICT_HORIZON
)

# 各个时间步的独立指标分析
print("\n" + "=" * 40)
print(f" ⏳ 各个预测步长（1~{PREDICT_HORIZON}分钟）的独立指标")
print("=" * 40)
for step in range(PREDICT_HORIZON):
    step_y_true = y_true_matrix[:, step]
    step_y_pred = y_pred_matrix[:, step]
    step_hit = industrial_hit_rate_pct(step_y_true, step_y_pred)
    step_mae = mean_absolute_error(step_y_true, step_y_pred)
    print(f"未来第 {step+1:2d} 分钟 | 命中率: {step_hit:6.2f}% | MAE: {step_mae:.4f}")
print("=" * 40 + "\n")

# ==========================================
# 6. 可视化
# ==========================================
ensure_output_dir()

# 绘制特征重要性
plt.figure(figsize=(10, 8))
# XGBoost 原生支持多输出后的特征重要性
feature_imp = pd.DataFrame(
    sorted(zip(model.feature_importances_, X_train.columns)),
    columns=["Value", "Feature"],
)
top_20 = feature_imp.tail(20)
plt.barh(top_20["Feature"], top_20["Value"], color="lightcoral")
plt.title(f"Top 20 Feature Importance (XGBoost Native Multi-output)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
imp_path = f"{OUTPUT_DIR}/xgboost_importance_multistep.png"
plt.savefig(imp_path, dpi=150)
plt.close()

# 选取【最远一步】(即第 H 步) 绘制预测曲线
y_true_last = y_true_matrix[:, -1][-PLOT_TAIL:]
y_pred_last = y_pred_matrix[:, -1][-PLOT_TAIL:]

plt.figure(figsize=(15, 5))
plt.plot(y_true_last, label=f"True Current (Step {PREDICT_HORIZON})", color="blue", alpha=0.6)
plt.plot(y_pred_last, label=f"XGBoost Pred (Step {PREDICT_HORIZON})", color="green", linestyle="--", alpha=0.8)
upper = y_true_last * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
lower = y_true_last * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
plt.fill_between(
    np.arange(len(y_true_last)), lower, upper, color="gray", alpha=0.15, 
    label=f"±{INDUSTRIAL_HIT_REL_TOLERANCE*100:g}% band"
)
plt.title(f"Ball Mill Current — XGBoost Native Multi-output (Horizon {PREDICT_HORIZON})")
plt.legend()
plt.grid(True)
plt.tight_layout()
pred_png = f"{OUTPUT_DIR}/xgboost_predict_multistep.png"
plt.savefig(pred_png, dpi=150)
plt.close()

print(f"特征重要性图: {imp_path}")
print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")