import warnings
import os
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

# 注意：这里导入了刚刚在 config 中新增的 create_sequences_multistep
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
    append_model_eval_report,
    create_sequences_multistep,
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
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. PCMP-Mamba 架构 (支持多步输出)
# ==========================================
class PhysicsGatedUnit(nn.Module):
    def __init__(self, d_model, phys_dim=3):
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)
        self.phys_proj = nn.Linear(phys_dim, d_model)

    def forward(self, x, phys_prior):
        g = torch.sigmoid(self.gate(x) + self.phys_proj(phys_prior))
        return x * g + x  

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
        out_fast = self.fast_ssm(x_fast)
        out_slow = self.slow_ssm(x_slow)
        
        feat_fast_last = out_fast[:, -1, :]  
        feat_slow_last = out_slow[:, -1, :]  
        
        w = torch.softmax(self.fuse_weight, dim=0)
        combined = w[0] * feat_fast_last + w[1] * feat_slow_last
        return self.norm(self.dropout(combined))

class PCMPMamba(nn.Module):
    # 【修改】：加入了 pred_len (预测步长)
    def __init__(self, seq_len, pred_len, fast_idx, slow_idx, target_feat_idx, d_model=128):
        super().__init__()
        self.fast_idx = fast_idx
        self.slow_idx = slow_idx
        self.target_feat_idx = target_feat_idx
        self.d_model = d_model
        
        self.fast_proj = nn.Linear(len(fast_idx), d_model)
        self.slow_proj = nn.Linear(len(slow_idx), d_model)
        
        self.hetero_block = HeteroMambaBlock(d_model)
        self.phys_gate = PhysicsGatedUnit(d_model, phys_dim=3)  
        
        # 【修改】：残差和趋势投射层的输出维度变为 pred_len，不再是 1
        self.residual_head = nn.Linear(d_model, pred_len)
        self.trend_head = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        B, L, V = x.shape
        curr_series = x[:, :, self.target_feat_idx]  
        
        diff_1 = curr_series[:, -1] - curr_series[:, -2]           
        last_val = curr_series[:, -1]                              
        local_mean = curr_series[:, -5:].mean(dim=1)               
        phys_prior = torch.stack([diff_1, last_val, local_mean], dim=1) 

        x_fast = x[:, :, self.fast_idx]  
        x_slow = x[:, :, self.slow_idx]  
        
        f_feat = self.fast_proj(x_fast) 
        s_feat = self.slow_proj(x_slow)
        
        combined_feat = self.hetero_block(f_feat, s_feat) 
        gated_feat = self.phys_gate(combined_feat, phys_prior) 
        
        # 输出形状将变为 (B, pred_len)
        res = self.residual_head(gated_feat)  
        trend = self.trend_head(curr_series)  
        
        return res + trend

# ==========================================
# 3. 数据处理 (使用多步预测采样)
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

# 【修改】：使用多步构造序列，y_seq 的形状将是 (N, PREDICT_HORIZON)
X_seq, y_seq = create_sequences_multistep(X_raw, y_raw, LAG_STEPS, PREDICT_HORIZON)

total_len = len(X_seq)
train_end = int(total_len * TRAIN_RATIO)
val_end = int(total_len * (TRAIN_RATIO + VAL_RATIO))

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=BATCH_SIZE)

# ==========================================
# 4. 训练模型
# ==========================================
model = PCMPMamba(
    seq_len=LAG_STEPS,
    pred_len=PREDICT_HORIZON,  # 传入预测步数
    fast_idx=FAST_COLS,
    slow_idx=SLOW_COLS,
    target_feat_idx=ti,
).to(DEVICE)

if torch.cuda.device_count() > 1:
    print(f"检测到 {torch.cuda.device_count()} 张显卡，使用 DataParallel 进行并行训练")
    model = nn.DataParallel(model) 
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.MSELoss() # PyTorch 的 MSE 自动支持计算 (Batch, pred_len) 多维矩阵的误差

print(f"\n开始训练连续 {PREDICT_HORIZON} 步 PCMP-Mamba...")
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
        torch.save(model.state_dict(), "best_pcmp_mamba_multistep.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("触发早停，结束训练")
            break

# ==========================================
# 5. 测试与指标提取
# ==========================================
model.load_state_dict(torch.load("best_pcmp_mamba_multistep.pth", map_location=DEVICE))
model.eval()

y_preds_scaled = []
y_true_scaled = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(DEVICE)
        preds = model(batch_x)
        y_preds_scaled.append(preds.cpu().numpy())
        y_true_scaled.append(batch_y.numpy())

# 将批次拼接，形成形状为 (N, PREDICT_HORIZON) 的矩阵
y_pred_matrix = np.concatenate(y_preds_scaled)
y_true_matrix = np.concatenate(y_true_scaled)

# 【修改】：反归一化需要把数据展平为一列，再复原成 (N, PREDICT_HORIZON) 的形状
y_pred_inv = scaler_y.inverse_transform(y_pred_matrix.reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)
y_true_inv = scaler_y.inverse_transform(y_true_matrix.reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)

# 评估：传入展平的数组，计算所有 10 步的全局平均误差性能
metrics = regression_metrics(y_true_inv.flatten(), y_pred_inv.flatten())
print_standard_eval_report(f"PCMP-Mamba (连续 {PREDICT_HORIZON} 步)", metrics)
append_model_eval_report(
    "PCMP-Mamba-MultiStep", metrics, 
    lag_steps=LAG_STEPS, predict_horizon=PREDICT_HORIZON
)

# ==========================================
# 6. 多步连续预测的图像展示
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 为了避免在同一个时间点画10条重叠线导致图像混乱
# 我们选择画出模型对于【预测序列最后一步】（即第H步）的预测，展示模型的极限预测能力
y_true_last_step = y_true_inv[:, -1][-PLOT_TAIL:]
y_pred_last_step = y_pred_inv[:, -1][-PLOT_TAIL:]

plt.figure(figsize=(15, 5))
plt.plot(y_true_last_step, label=f"True Current (Step {PREDICT_HORIZON})", color="blue", alpha=0.6)
plt.plot(y_pred_last_step, label=f"Predicted Current (Step {PREDICT_HORIZON})", color="red", linestyle="--", alpha=0.8)
upper = y_true_last_step * (1.0 + INDUSTRIAL_HIT_REL_TOLERANCE)
lower = y_true_last_step * (1.0 - INDUSTRIAL_HIT_REL_TOLERANCE)
plt.fill_between(
    np.arange(len(y_true_last_step)), lower, upper, color="gray", alpha=0.15,
    label=f"±{INDUSTRIAL_HIT_REL_TOLERANCE*100:g}% band",
)
plt.title(f"Ball Mill Current — PCMP-Mamba (Horizon {PREDICT_HORIZON}, Last {PLOT_TAIL} Windows)")
plt.legend()
plt.grid(True)
plt.tight_layout()
pred_png = f"{OUTPUT_DIR}/pcmp_mamba_predict_multistep.png"
plt.savefig(pred_png, dpi=150)
plt.close()

# 绘制这 H 步所有的误差分布
plt.figure(figsize=(10, 5))
plt.hist((y_true_inv - y_pred_inv).flatten(), bins=50, color="#2ca02c", alpha=0.7, edgecolor="black")
plt.title(f"Prediction Error Distribution (All {PREDICT_HORIZON} Steps)")
plt.xlabel("Error (A)")
plt.ylabel("Frequency")
plt.axvline(0, color="red", linestyle="dashed", linewidth=1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pcmp_mamba_error_hist_multistep.png", dpi=120)
plt.close()

print(f"预测曲线已保存: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")

try:
    del model, optimizer, train_loader, val_loader, test_loader, criterion
except NameError:
    pass
release_gpu_memory()