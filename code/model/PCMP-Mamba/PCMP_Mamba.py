import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mamba_ssm import Mamba

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
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. PCMP-Mamba 架构
# ==========================================
class PhysicsGatedUnit(nn.Module):
    def __init__(self, d_model, phys_dim=3):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)
        # 物理先验维度从 1 扩展到 phys_dim (如3)
        self.phys_proj = nn.Linear(phys_dim, d_model)

    def forward(self, x, phys_prior):
        # x: (B, d_model), phys_prior: (B, phys_dim)
        g = torch.sigmoid(self.gate(x) + self.phys_proj(phys_prior))
        return x * g + x  # 加入残差连接，防止门控过度抑制

class HeteroMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        # fast_ssm 处理振动等高频信号，d_conv较小，感受野集中在短期突变
        self.fast_ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=2)
        # slow_ssm 处理温度等低频信号，d_conv较大，感受野更宽，捕捉平滑趋势
        self.slow_ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=7)
        
        self.fuse_weight = nn.Parameter(torch.ones(2))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_fast, x_slow):
        # Mamba 的正确输入形状应当是: (B, L, d_model)
        out_fast = self.fast_ssm(x_fast)
        out_slow = self.slow_ssm(x_slow)
        
        # 提取最后一个时间步的特征 (替代 mean)，因为时序预测最新状态最重要
        feat_fast_last = out_fast[:, -1, :]  # (B, d_model)
        feat_slow_last = out_slow[:, -1, :]  # (B, d_model)
        
        w = torch.softmax(self.fuse_weight, dim=0)
        combined = w[0] * feat_fast_last + w[1] * feat_slow_last
        
        return self.norm(self.dropout(combined))

class PCMPMamba(nn.Module):
    def __init__(self, seq_len, fast_idx, slow_idx, target_feat_idx, d_model=128):
        super().__init__()
        self.fast_idx = fast_idx
        self.slow_idx = slow_idx
        self.target_feat_idx = target_feat_idx
        self.d_model = d_model
        
        # 【修改关键】投影是对特征维度(Feature)做投影，而不是序列长度(Time)
        self.fast_proj = nn.Linear(len(fast_idx), d_model)
        self.slow_proj = nn.Linear(len(slow_idx), d_model)
        
        self.hetero_block = HeteroMambaBlock(d_model)
        self.phys_gate = PhysicsGatedUnit(d_model, phys_dim=3)  # 增强物理先验
        
        self.residual_head = nn.Linear(d_model, 1)
        self.trend_head = nn.Linear(seq_len, 1)

    def forward(self, x):
        # x shape: (B, L, V)
        B, L, V = x.shape
        curr_series = x[:, :, self.target_feat_idx]  # (B, L)
        
        # 【物理先验增强】构建更丰富的物理状态: [一阶差分, 最后一刻绝对值, 局部均值]
        diff_1 = curr_series[:, -1] - curr_series[:, -2]           # 短期波动
        last_val = curr_series[:, -1]                              # 当前载荷状态
        local_mean = curr_series[:, -5:].mean(dim=1)               # 5分钟平滑载荷
        phys_prior = torch.stack([diff_1, last_val, local_mean], dim=1) # (B, 3)

        # 提取快慢特征，并投射到 d_model
        x_fast = x[:, :, self.fast_idx]  # (B, L, V_fast)
        x_slow = x[:, :, self.slow_idx]  # (B, L, V_slow)
        
        # 此时 f_feat 形状为 (B, L, d_model)，完美符合 Mamba 对时序的要求
        f_feat = self.fast_proj(x_fast) 
        s_feat = self.slow_proj(x_slow)
        
        # 经过 Mamba 处理并融合
        combined_feat = self.hetero_block(f_feat, s_feat) # (B, d_model)
        
        # 物理门控调节
        gated_feat = self.phys_gate(combined_feat, phys_prior) # (B, d_model)
        
        # 预测：残差部分(基于多变量 Mamba) + 趋势部分(基于自回归)
        res = self.residual_head(gated_feat)  # (B, 1)
        trend = self.trend_head(curr_series)  # (B, 1)
        
        return res + trend

# ==========================================
# 3. 数据
# ==========================================
print(f"加载数据 | 设备: {DEVICE}")
df = pd.read_csv(FILE_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df.set_index("time", inplace=True)
df = df.ffill().bfill()

print("正在进行工业数据去噪...")
df = NR.clean_industrial_data(df)

cols = list(df.columns)
ti = cols.index(TARGET_COL)
FAST_COLS = [ti] + [cols.index(c) for c in cols if "Vibrate" in c]
SLOW_COLS = [i for i in range(len(cols)) if i not in FAST_COLS]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_raw = scaler_x.fit_transform(df.values)
y_raw = scaler_y.fit_transform(df[[TARGET_COL]].values)

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
# 4. 训练
# ==========================================
model = PCMPMamba(
    seq_len=LAG_STEPS,
    fast_idx=FAST_COLS,
    slow_idx=SLOW_COLS,
    target_feat_idx=ti,
).to(DEVICE)

# --- 修改这里 ---
if torch.cuda.device_count() > 1:
    print(f"检测到 {torch.cuda.device_count()} 张显卡，使用 DataParallel 进行并行训练")
    model = nn.DataParallel(model) 

model = model.to(DEVICE)
# ----------------

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.MSELoss()

print("\n开始训练 PCMP-Mamba...")
best_val_loss = float("inf")
patience = 10
counter = 0

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    model.eval()
    val_l = 0
    with torch.no_grad():
        for bx, by in val_loader:
            val_l += criterion(model(bx.to(DEVICE)), by.to(DEVICE)).item()
    avg_val = val_l / len(val_loader)
    print(f"Validation Loss: {avg_val:.6f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "best_pcmp_mamba.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("早停")
            break

# ==========================================
# 5. 测试与指标
# ==========================================
model.load_state_dict(torch.load("best_pcmp_mamba.pth", map_location=DEVICE))
model.eval()

y_preds_scaled = []
y_true_scaled = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        preds = model(batch_x)
        y_preds_scaled.append(preds.cpu().numpy())
        y_true_scaled.append(batch_y.numpy())

y_pred = scaler_y.inverse_transform(np.concatenate(y_preds_scaled)).flatten()
y_true = scaler_y.inverse_transform(np.concatenate(y_true_scaled)).flatten()

metrics = regression_metrics(y_true, y_pred)
print_standard_eval_report("PCMP-Mamba", metrics)
append_model_eval_report(
    "PCMP-Mamba",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

pred_png = save_prediction_plot(
    y_true,
    y_pred,
    out_filename="pcmp_mamba_predict.png",
    title=f"Ball Mill Current — PCMP-Mamba (H={PREDICT_HORIZON}, last {PLOT_TAIL})",
    pred_label="PCMP-Mamba Pred",
)
print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")

plt.figure(figsize=(10, 5))
plt.hist(y_true - y_pred, bins=50, color="#2ca02c", alpha=0.7, edgecolor="black")
plt.title("Prediction Error Distribution")
plt.xlabel("Error (A)")
plt.ylabel("Frequency")
plt.axvline(0, color="red", linestyle="dashed", linewidth=1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pcmp_mamba_error_hist.png", dpi=120)
plt.close()
