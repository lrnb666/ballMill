import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from Aprocess import noiseReduce as NR

# 【修改 1】：导入连续多步构造函数 create_sequences_multistep 和 工业命中率函数
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
    create_sequences_multistep,  # <--- 使用多步的函数
    print_standard_eval_report,
    regression_metrics,
    release_gpu_memory,
)

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置
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
# 【修改 2】：使用多步构造序列，此时 y_seq 的形状为 (样本数, PREDICT_HORIZON)
X_seq, y_seq = create_sequences_multistep(X_raw, y_raw, LAG_STEPS, PREDICT_HORIZON)

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
        # 取最后一个时间步的隐藏状态，通过全连接层输出 output_size 个连续预测值
        return self.fc(out[:, -1, :])


# 【修改 3】：将 output_size 设为 PREDICT_HORIZON (即多步长度)
model = BallMillGRU(
    input_size=X_raw.shape[1], 
    hidden_size=64, 
    num_layers=2, 
    output_size=PREDICT_HORIZON 
)

if torch.cuda.device_count() > 1:
    print(f"🔥 检测到 {torch.cuda.device_count()} 张显卡，启用多卡 DataParallel 并行训练！")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. 训练
# ==========================================
print(f"\n开始训练连续 {PREDICT_HORIZON} 步 GRU 模型...")
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
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_gru_model_multistep.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"早停触发于第 {epoch+1} 轮")
            break

# ==========================================
# 6. 评估与反归一化
# ==========================================
model.load_state_dict(torch.load("best_gru_model_multistep.pth", map_location=DEVICE))
model.eval()

y_preds_scaled = []
y_true_scaled = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        preds = model(batch_x)
        y_preds_scaled.append(preds.cpu().numpy())
        y_true_scaled.append(batch_y.numpy())

# 【修改 4】：处理多维数组并反归一化
y_pred_matrix = np.concatenate(y_preds_scaled)
y_true_matrix = np.concatenate(y_true_scaled)

# 展平 -> 反归一化 -> 重新塑形回 (样本数, PREDICT_HORIZON)
y_pred_inv = scaler_y.inverse_transform(y_pred_matrix.reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)
y_true_inv = scaler_y.inverse_transform(y_true_matrix.reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)

# 评估整体性能 (将所有预测点展平进行评估)
metrics = regression_metrics(y_true_inv.flatten(), y_pred_inv.flatten())
print_standard_eval_report(f"GRU (连续 {PREDICT_HORIZON} 步整体平均)", metrics)
append_model_eval_report(
    "GRU-MultiStep", metrics, lag_steps=LAG_STEPS, predict_horizon=PREDICT_HORIZON,
)

# 【附加功能】：查看各个步长的衰减性能 (非常适合时序预测分析)
print("\n" + "=" * 40)
print(f" ⏳ 各个预测步长（1~{PREDICT_HORIZON}分钟）的独立指标衰减")
print("=" * 40)
for step in range(PREDICT_HORIZON):
    step_y_true = y_true_inv[:, step]
    step_y_pred = y_pred_inv[:, step]
    step_hit = industrial_hit_rate_pct(step_y_true, step_y_pred)
    step_mae = mean_absolute_error(step_y_true, step_y_pred)
    print(f"未来第 {step+1:2d} 分钟 | 命中率: {step_hit:6.2f}% | MAE: {step_mae:.4f}")
print("=" * 40 + "\n")


# ==========================================
# 7. 可视化（训练曲线 + 预测）
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loss 曲线
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.title(f"GRU Multi-step Loss (H={PREDICT_HORIZON})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/gru_loss_curve_multistep.png", dpi=120)
plt.close()

# 【修改 5】：绘制预测曲线时，我们选取【最后一步】进行绘制，代表模型最远的预测能力
y_true_last = y_true_inv[:, -1][-PLOT_TAIL:]
y_pred_last = y_pred_inv[:, -1][-PLOT_TAIL:]

plt.figure(figsize=(15, 5))
plt.plot(y_true_last, label=f"True Current (Step {PREDICT_HORIZON})", color="blue", alpha=0.6)
plt.plot(y_pred_last, label=f"GRU Pred (Step {PREDICT_HORIZON})", color="red", linestyle="--", alpha=0.8)
upper = y_true_last * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
lower = y_true_last * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
plt.fill_between(
    np.arange(len(y_true_last)), lower, upper, color="gray", alpha=0.15, 
    label=f"±{INDUSTRIAL_HIT_REL_TOLERANCE*100:g}% band"
)
plt.title(f"Ball Mill Current — GRU (Horizon {PREDICT_HORIZON}, last {PLOT_TAIL})")
plt.legend()
plt.grid(True)
plt.tight_layout()
pred_png = f"{OUTPUT_DIR}/gru_predict_multistep.png"
plt.savefig(pred_png, dpi=150)
plt.close()

print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")

try:
    del model, optimizer, train_loader, val_loader, test_loader, criterion
except NameError:
    pass
release_gpu_memory()