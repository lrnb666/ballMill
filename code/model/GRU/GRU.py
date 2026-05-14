import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

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
    create_sequences,
    print_standard_eval_report,
    regression_metrics,
    save_prediction_plot,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置（见 Autils/eval_config）
# ==========================================
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据加载与预处理
# ==========================================
print(f"正在加载全量数据，使用设备: {DEVICE}")
df = pd.read_csv(FILE_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df.set_index("time", inplace=True)
df = df.ffill().bfill()

print("正在进行工业数据去噪...")
df = NR.clean_industrial_data(df)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_raw = scaler_x.fit_transform(df.values)
y_raw = scaler_y.fit_transform(df[[TARGET_COL]].values)
print(f"特征总数: {X_raw.shape[1]}")

# ==========================================
# 3. 构造序列
# ==========================================
X_seq, y_seq = create_sequences(X_raw, y_raw, LAG_STEPS, PREDICT_HORIZON)

total_len = len(X_seq)
train_end = int(total_len * TRAIN_RATIO)
val_end = int(total_len * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
    batch_size=BATCH_SIZE,
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
    batch_size=BATCH_SIZE,
)

# ==========================================
# 4. GRU 模型
# ==========================================
class BallMillGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# 修改后的多卡代码
model = BallMillGRU(
    input_size=X_raw.shape[1], hidden_size=64, num_layers=2, output_size=1
)

# 核心修改：检测显卡数量，如果大于1张，就用 DataParallel 包装模型
if torch.cuda.device_count() > 1:
    print(f"🔥 检测到 {torch.cuda.device_count()} 张显卡，启用多卡 DataParallel 并行训练！")
    model = nn.DataParallel(model)

# 最后再把模型推到 DEVICE
model = model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. 训练
# ==========================================
print("\n开始训练 GRU 模型...")
best_val_loss = float("inf")
patience = 7
counter = 0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            output = model(batch_x)
            total_val_loss += criterion(output, batch_y).item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} "
            f"Val Loss: {avg_val_loss:.6f}"
        )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_gru_model.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"早停触发于第 {epoch+1} 轮")
            break

# ==========================================
# 6. 评估
# ==========================================
model.load_state_dict(torch.load("best_gru_model.pth", map_location=DEVICE))
model.eval()
y_preds_scaled = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(DEVICE)
        preds = model(batch_x)
        y_preds_scaled.append(preds.cpu().numpy())

y_pred = scaler_y.inverse_transform(np.concatenate(y_preds_scaled)).flatten()
y_true = scaler_y.inverse_transform(y_test).flatten()

metrics = regression_metrics(y_true, y_pred)
print_standard_eval_report("GRU", metrics)
append_model_eval_report(
    "GRU",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

# ==========================================
# 7. 可视化（训练曲线 + 预测）
# ==========================================
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.title("GRU loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gru_loss_curve.png", dpi=120)
plt.close()

pred_png = save_prediction_plot(
    y_true,
    y_pred,
    out_filename="gru_predict.png",
    title=f"Ball Mill Current — GRU (H={PREDICT_HORIZON}, last {PLOT_TAIL})",
    pred_label="GRU Pred",
)
print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")
