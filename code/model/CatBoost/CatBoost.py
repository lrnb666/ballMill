import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

# 你原来的导入不动
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
# 2. 特征工程
# ==========================================
print(f"正在构造特征：过去 {LAG_STEPS} 分钟 → 未来连续 {PREDICT_HORIZON} 分钟...")
df_processed, target_cols = create_lag_features_df_multistep(
    df, TARGET_COL, LAG_STEPS, PREDICT_HORIZON
)
print(f"特征构造完成，目标列数: {len(target_cols)}")

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

# ==========================================
# 4. ✅ CatBoost 多卡训练（你有 2 张 GPU，直接用 0,1）
# ==========================================
print(f"\n🔥 启用 2 张 GPU 训练 CatBoost 多目标模型 (MultiRMSE)...")

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3,
    loss_function='MultiRMSE',
    eval_metric='MultiRMSE',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,

    # ========================
    # 多卡核心配置（已帮你写好）
    # ========================
    task_type='GPU',          # 启用 GPU
    devices='0:1',            # 直接使用你 2 张显卡
)

# 用 Pool 格式多卡训练更稳定（CatBoost官方推荐）
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

model.fit(
    train_pool,
    eval_set=val_pool,
    use_best_model=True
)

print(f"✅ CatBoost 2 卡多目标模型训练完成！")

# ==========================================
# 5. 评估（完全不动）
# ==========================================
print("\n在测试集上进行全序列预测与评估...")
y_pred_matrix = model.predict(X_test)
y_true_matrix = y_test.values

metrics = regression_metrics(y_true_matrix.flatten(), y_pred_matrix.flatten())
print_standard_eval_report(f"CatBoost-MultiRMSE (连续 {PREDICT_HORIZON} 步)", metrics)

append_model_eval_report(
    "CatBoost-Native-MultiOutput", metrics, 
    lag_steps=LAG_STEPS, predict_horizon=PREDICT_HORIZON
)

print("\n" + "=" * 40)
print(f" 各预测步长（1~{PREDICT_HORIZON}分钟）指标衰减")
print("=" * 40)
for step in range(PREDICT_HORIZON):
    yt = y_true_matrix[:, step]
    yp = y_pred_matrix[:, step]
    hit = industrial_hit_rate_pct(yt, yp)
    mae = mean_absolute_error(yt, yp)
    print(f"未来第 {step+1:2d} 分钟 | 命中率: {hit:6.2f}% | MAE: {mae:.4f}")
print("=" * 40 + "\n")

# ==========================================
# 6. 可视化（完全不动）
# ==========================================
ensure_output_dir()

feat_importance = model.get_feature_importance()
sorted_idx = np.argsort(feat_importance)
plt.figure(figsize=(10, 8))
plt.barh(X_train.columns[sorted_idx][-20:], feat_importance[sorted_idx][-20:], color='teal')
plt.title("Top 20 Features (CatBoost MultiRMSE)")
plt.xlabel("Feature Importance")
plt.tight_layout()
imp_path = f"{OUTPUT_DIR}/catboost_importance.png"
plt.savefig(imp_path, dpi=150)
plt.close()

y_true_last = y_true_matrix[:, -1][-PLOT_TAIL:]
y_pred_last = y_pred_matrix[:, -1][-PLOT_TAIL:]

plt.figure(figsize=(15, 5))
plt.plot(y_true_last, label=f"True (Step {PREDICT_HORIZON})", color="blue", alpha=0.6)
plt.plot(y_pred_last, label=f"CatBoost Pred (Step {PREDICT_HORIZON})", color="green", linestyle="--")

upper = y_true_last * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
lower = y_true_last * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
plt.fill_between(np.arange(len(y_true_last)), lower, upper, color="gray", alpha=0.15)

plt.title(f"Ball Mill Current — CatBoost Native (Horizon {PREDICT_HORIZON})")
plt.legend()
plt.grid(True)
plt.tight_layout()
pred_png = f"{OUTPUT_DIR}/catboost_predict.png"
plt.savefig(pred_png, dpi=150)
plt.close()

print(f"特征重要性图: {imp_path}")
print(f"预测趋势图: {pred_png}")
print(f"指标已追加至: {METRICS_REPORT_PATH}")
