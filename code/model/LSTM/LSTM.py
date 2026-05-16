import os
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
    create_sequences_multistep,
    print_standard_eval_report,
    regression_metrics,
    release_gpu_memory,
    save_prediction_plot,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置（见 Autils/eval_config；多卡可自行 export CUDA_VISIBLE_DEVICES）
# ==========================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()

# ==========================================
# 2. 数据加载与预处理
# ==========================================
print(f"正在加载数据 | 设备: {DEVICE} | GPU 数: {GPU_COUNT}")
df = pd.read_csv(FILE_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df.set_index("time", inplace=True)
df = df.ffill().bfill()

print("正在进行工业数据去噪...")
df = NR.clean_industrial_data(df)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_data = scaler_x.fit_transform(df.values)
y_data = scaler_y.fit_transform(df[[TARGET_COL]].values)

# ==========================================
# 3. 序列
# ==========================================
X_seq, y_seq = create_sequences_multistep(X_data, y_data, LAG_STEPS, PREDICT_HORIZON)

total_len = len(X_seq)
train_end = int(total_len * TRAIN_RATIO)
val_end = int(total_len * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

nw = min(4, os.cpu_count() or 1)
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=nw,
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
# 4. 模型
# ==========================================
class BallMillLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


raw_model = BallMillLSTM(
    input_size=X_data.shape[1],
    hidden_size=128,
    num_layers=2,
    output_size=PREDICT_HORIZON,
)
if GPU_COUNT > 1:
    print("启用 DataParallel")
    model = nn.DataParallel(raw_model)
else:
    model = raw_model

model = model.to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. 训练
# ==========================================
print(f"\n开始训练 LSTM（连续 {PREDICT_HORIZON} 步）...")
best_val_loss = float("inf")
patience = 7
counter = 0
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    t_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()

    model.eval()
    v_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            v_loss += criterion(model(batch_x), batch_y).item()

    avg_t = t_loss / len(train_loader)
    avg_v = v_loss / len(val_loader)
    train_losses.append(avg_t)
    val_losses.append(avg_v)
    print(f"Epoch {epoch+1:02d} | Train: {avg_t:.6f} | Val: {avg_v:.6f}")

    if avg_v < best_val_loss:
        best_val_loss = avg_v
        state_dict = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
        torch.save(state_dict, "best_lstm_multistep.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("早停。")
            break

# ==========================================
# 6. 评估
# ==========================================
checkpoint = torch.load("best_lstm_multistep.pth", map_location=DEVICE)
if GPU_COUNT > 1:
    model.module.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.eval()
y_preds_scaled = []
y_true_scaled = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        y_preds_scaled.append(model(batch_x).cpu().numpy())
        y_true_scaled.append(batch_y.numpy())

y_pred_matrix = np.concatenate(y_preds_scaled)
y_true_matrix = np.concatenate(y_true_scaled)
y_pred_inv = scaler_y.inverse_transform(y_pred_matrix.reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)
y_true_inv = scaler_y.inverse_transform(y_true_matrix.reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)

metrics = regression_metrics(y_true_inv.flatten(), y_pred_inv.flatten())
print_standard_eval_report(f"LSTM（连续 {PREDICT_HORIZON} 步）", metrics)
append_model_eval_report(
    "LSTM-MultiStep",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.title(f"LSTM loss (H={PREDICT_HORIZON})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/lstm_loss_curve.png", dpi=120)
plt.close()

pred_png = save_prediction_plot(
    y_true_inv[:, -1],
    y_pred_inv[:, -1],
    out_filename="lstm_predict_multistep.png",
    title=f"Ball Mill Current — LSTM step {PREDICT_HORIZON} (last {PLOT_TAIL})",
    pred_label=f"LSTM Pred (step {PREDICT_HORIZON})",
)
print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")

try:
    del model, optimizer, train_loader, val_loader, test_loader, criterion, checkpoint
except NameError:
    pass
release_gpu_memory()
