import warnings

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

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
    xgb_preferred_device,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置（路径/窗口/预测步长见 Autils/eval_config）
# ==========================================
_xgb_device = xgb_preferred_device()

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
# 3. 特征工程
# ==========================================
df_processed = create_lag_features_df(df, TARGET_COL, LAG_STEPS, PREDICT_HORIZON)

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

# ==========================================
# 5. XGBoost 训练
# ==========================================
print("\n开始训练 XGBoost 模型...")
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    device=_xgb_device,
    early_stopping_rounds=50,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# ==========================================
# 6. 模型评估
# ==========================================
print("\n在测试集上进行评估...")
y_pred = model.predict(X_test)

metrics = regression_metrics(y_test, y_pred)
print_standard_eval_report("XGBoost", metrics)
append_model_eval_report(
    "XGBoost",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

# ==========================================
# 7. 可视化
# ==========================================
ensure_output_dir()
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(
    model,
    max_num_features=20,
    importance_type="gain",
    title="XGBoost Feature Importance",
    ax=ax,
    color="coral",
)
imp_path = f"{OUTPUT_DIR}/xgboost_importance.png"
plt.tight_layout()
plt.savefig(imp_path, dpi=150)
plt.close()
print(f"特征重要性图: {imp_path}")

pred_png = save_prediction_plot(
    y_test,
    y_pred,
    out_filename="xgboost_predict.png",
    title=f"Current Prediction (H={PREDICT_HORIZON}) — last {PLOT_TAIL} points",
    pred_label="XGBoost Pred",
)
print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")
