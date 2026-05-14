import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

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
    append_model_eval_report,
    create_lag_features_df,
    ensure_output_dir,
    print_standard_eval_report,
    regression_metrics,
    save_prediction_plot,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置（见 Autils/eval_config）
# ==========================================

# ==========================================
# 2. 数据加载与预处理
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
# 3. 特征工程
# ==========================================
print(f"正在构造特征：过去 {LAG_STEPS} 分钟 → 未来第 {PREDICT_HORIZON} 分钟...")
df_processed = create_lag_features_df(df, TARGET_COL, LAG_STEPS, PREDICT_HORIZON)
print(f"特征构造完成，特征维度：{df_processed.shape[1] - 1} 列。")

# ==========================================
# 4. 数据集划分
# ==========================================
total_samples = len(df_processed)
train_size = int(total_samples * TRAIN_RATIO)
val_size = int(total_samples * VAL_RATIO)

train_data = df_processed.iloc[:train_size]
val_data = df_processed.iloc[train_size : train_size + val_size]
test_data = df_processed.iloc[train_size + val_size :]

X_train, y_train = train_data.drop("target", axis=1), train_data["target"]
X_val, y_val = val_data.drop("target", axis=1), val_data["target"]
X_test, y_test = test_data.drop("target", axis=1), test_data["target"]
print(f"训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}")

# ==========================================
# 5. LightGBM 训练
# ==========================================
print("\n开始训练 LightGBM 模型...")
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=callbacks,
)

# ==========================================
# 6. 评估
# ==========================================
print("\n在测试集上进行预测与评估...")
y_pred = model.predict(X_test)
metrics = regression_metrics(y_test, y_pred)
print_standard_eval_report("LightGBM", metrics)
append_model_eval_report(
    "LightGBM",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

# ==========================================
# 7. 可视化
# ==========================================
ensure_output_dir()
plt.figure(figsize=(10, 8))
feature_imp = pd.DataFrame(
    sorted(zip(model.feature_importances_, X_train.columns)),
    columns=["Value", "Feature"],
)
top_20 = feature_imp.tail(20)
plt.barh(top_20["Feature"], top_20["Value"], color="skyblue")
plt.title("Top 20 Feature Importance (Gain)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
imp_path = f"{OUTPUT_DIR}/lightgbm_importance.png"
plt.savefig(imp_path, dpi=150)
plt.close()
print(f"特征重要性图: {imp_path}")

pred_png = save_prediction_plot(
    y_test,
    y_pred,
    out_filename="lightgbm_predict.png",
    title=f"Current Prediction (H={PREDICT_HORIZON}) — last {PLOT_TAIL} points",
    pred_label="LightGBM Pred",
)
print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")
