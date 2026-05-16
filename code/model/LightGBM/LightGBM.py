import warnings
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from Aprocess import noiseReduce as NR

# 【修改 1】：引入必要的评估与配置
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

# X 包含所有的 lag 特征，y 包含所有的 target_step_x 列
X_train, y_train = train_data.drop(columns=target_cols), train_data[target_cols]
X_val, y_val = val_data.drop(columns=target_cols), val_data[target_cols]
X_test, y_test = test_data.drop(columns=target_cols), test_data[target_cols]

print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# ==========================================
# 4. LightGBM 训练 (多步循环预测策略)
# ==========================================
print("\n开始训练 LightGBM 多步预测模型...")

models = []
y_preds_list = []
feature_importances = np.zeros(X_train.shape[1])

# 因为 LightGBM 原生不支持多目标，我们训练 H 个模型分别预测未来第 i 步
for step, t_col in enumerate(target_cols, start=1):
    print(f"\n--- 🚀 正在训练第 {step}/{PREDICT_HORIZON} 步预测模型 ({t_col}) ---")
    
    # 取出当前步的标签
    y_train_step = y_train[t_col]
    y_val_step = y_val[t_col]
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    # early_stopping 需要单独为每一步配置
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
    model.fit(
        X_train, y_train_step,
        eval_set=[(X_val, y_val_step)],
        eval_metric="rmse",
        callbacks=callbacks,
    )
    
    models.append(model)
    # 累加特征重要性以供后续平均分析
    feature_importances += model.feature_importances_
    
    # 预测测试集的当前步
    y_pred_step = model.predict(X_test)
    y_preds_list.append(y_pred_step)
    
    print(f"✅ 第 {step} 步模型训练完成，最佳迭代次数: {model.best_iteration_}")

# ==========================================
# 5. 评估
# ==========================================
print("\n在测试集上进行全序列预测与评估...")

# 将 H 个 (N,) 的预测结果按列拼成矩阵 (N, H)
# 注意：树模型通常不需要 StandardScaler，所以这里的数据已经是反归一化的原值了
y_pred_matrix = np.column_stack(y_preds_list)
y_true_matrix = y_test.values

# 全局平均指标计算
metrics = regression_metrics(y_true_matrix.flatten(), y_pred_matrix.flatten())
print_standard_eval_report(f"LightGBM (连续 {PREDICT_HORIZON} 步整体平均)", metrics)
append_model_eval_report(
    "LightGBM-MultiStep", metrics, 
    lag_steps=LAG_STEPS, predict_horizon=PREDICT_HORIZON
)

# 各个时间步的独立指标衰减分析
print("\n" + "=" * 40)
print(f" ⏳ 各个预测步长（1~{PREDICT_HORIZON}分钟）的独立指标衰减")
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

# 绘制全局平均特征重要性 (融合了 H 步的影响力)
avg_feature_importances = feature_importances / PREDICT_HORIZON
plt.figure(figsize=(10, 8))
feature_imp = pd.DataFrame(
    sorted(zip(avg_feature_importances, X_train.columns)),
    columns=["Value", "Feature"],
)
top_20 = feature_imp.tail(20)
plt.barh(top_20["Feature"], top_20["Value"], color="skyblue")
plt.title(f"Top 20 Averaged Feature Importance (Across {PREDICT_HORIZON} steps)")
plt.xlabel("Average Importance Score")
plt.ylabel("Features")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
imp_path = f"{OUTPUT_DIR}/lightgbm_importance_multistep.png"
plt.savefig(imp_path, dpi=150)
plt.close()
print(f"特征重要性图: {imp_path}")

# 选取【最远一步】(即第 H 步) 绘制测试集末尾的预测曲线
y_true_last = y_true_matrix[:, -1][-PLOT_TAIL:]
y_pred_last = y_pred_matrix[:, -1][-PLOT_TAIL:]

plt.figure(figsize=(15, 5))
plt.plot(y_true_last, label=f"True Current (Step {PREDICT_HORIZON})", color="blue", alpha=0.6)
plt.plot(y_pred_last, label=f"LightGBM Pred (Step {PREDICT_HORIZON})", color="red", linestyle="--", alpha=0.8)
upper = y_true_last * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
lower = y_true_last * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
plt.fill_between(
    np.arange(len(y_true_last)), lower, upper, color="gray", alpha=0.15, 
    label=f"±{INDUSTRIAL_HIT_REL_TOLERANCE*100:g}% band"
)
plt.title(f"Ball Mill Current — LightGBM (Horizon {PREDICT_HORIZON}, last {PLOT_TAIL})")
plt.legend()
plt.grid(True)
plt.tight_layout()
pred_png = f"{OUTPUT_DIR}/lightgbm_predict_multistep.png"
plt.savefig(pred_png, dpi=150)
plt.close()

print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")

try:
    del models
except NameError:
    pass
release_gpu_memory()