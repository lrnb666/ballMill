import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Aprocess import noiseReduce as NR

# 从你自己的 config 中导入所需的所有配置和方法
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
    fit_minmax_scalers_train_only,
    make_dataloader_generator,
    print_standard_eval_report,
    regression_metrics,
    release_gpu_memory,
    save_multistep_horizon_plot,
    set_global_seed,
)

warnings.filterwarnings("ignore")
set_global_seed()

# ==========================================
# 1. 配置
# ==========================================
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()

# ==========================================
# 2. 标准 PatchTST 架构 (纯 PyTorch 实现)
# ==========================================
class RevIN(nn.Module):
    """可逆实例归一化 (Reversible Instance Normalization) - 解决时序数据分布偏移的核心组件"""
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # 针对每个通道的可学习仿射变换参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            # x shape: [Batch, Seq_len, Vars]
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = x - self.mean
            x = x / self.stdev
            x = x * self.gamma + self.beta
            return x
        elif mode == 'denorm':
            # x shape: [Batch, Pred_len, Vars]
            x = (x - self.beta) / self.gamma
            x = x * self.stdev + self.mean
            return x

class PatchTST(nn.Module):
    def __init__(
        self, 
        seq_len, 
        pred_len, 
        num_vars,
        target_idx, 
        patch_len=16, 
        stride=8, 
        d_model=128, 
        n_heads=8, 
        e_layers=3, 
        dropout=0.2
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_vars = num_vars
        self.target_idx = target_idx
        
        # 保护机制: 确保 patch_len 和 stride 不会大于 seq_len
        self.patch_len = min(patch_len, seq_len)
        self.stride = min(stride, self.patch_len)
        self.patch_num = int((seq_len - self.patch_len) / self.stride + 1)
        
        # 1. RevIN 模块
        self.revin = RevIN(num_vars)
        
        # 2. 共享的 Token 映射层 (所有通道共享)
        self.value_embedding = nn.Linear(self.patch_len, d_model)
        
        # 3. 共享的位置编码 (所有通道共享)
        self.position_embedding = nn.Parameter(torch.randn(1, self.patch_num, d_model) * 0.02)
        
        # 4. 共享的 Transformer 编码器 (所有通道共享权重)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.dropout = nn.Dropout(dropout)
        
        # 5. 每个通道独立的预测头 (Independent Heads per Channel)
        # 为每个变量生成一个专属的 Linear 映射层
        self.heads = nn.ModuleList([
            nn.Linear(self.patch_num * d_model, pred_len) for _ in range(num_vars)
        ])

    def forward(self, x):
        # 1. 输入数据 x 形状: [Batch, Seq_len, Vars]
        B, L, V = x.shape
        
        # 2. RevIN 归一化
        x = self.revin(x, mode='norm')
        
        # 3. 转置以切分 Patch: [Batch, Vars, Seq_len]
        x = x.transpose(1, 2)
        
        # 使用 unfold 切分 Patch: [Batch, Vars, Patch_num, Patch_len]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 将 Vars 融合进 Batch 维度，以便所有变量共享同一个 Transformer 进行独立前向传播
        # 形状变为: [B * V, Patch_num, Patch_len]
        patches = patches.reshape(B * V, self.patch_num, self.patch_len)
        
        # 4. 映射与位置编码
        x_emb = self.value_embedding(patches) + self.position_embedding
        x_emb = self.dropout(x_emb)
        
        # 5. Transformer 处理: [B * V, Patch_num, d_model]
        enc_out = self.encoder(x_emb)
        
        # 6. 准备进入预测头，恢复变量维度: [B, V, Patch_num * d_model]
        enc_out = enc_out.reshape(B, V, -1)
        
        # 7. 每个变量使用各自独立的 Head 进行预测
        out = torch.zeros(B, V, self.pred_len, device=x.device)
        for i in range(V):
            out[:, i, :] = self.heads[i](enc_out[:, i, :])
            
        # out 形状恢复为 [Batch, Pred_len, Vars]，以匹配 RevIN 的 denorm 格式
        out = out.transpose(1, 2)
        
        # 8. RevIN 逆归一化
        out = self.revin(out, mode='denorm')
        
        # 9. 返回目标列的预测结果: [Batch, Pred_len]
        return out[:, :, self.target_idx]


# ==========================================
# 3. 数据处理
# ==========================================
print(f"加载数据 | 设备: {DEVICE} | GPU 数: {GPU_COUNT}")
df = pd.read_csv(FILE_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df.set_index("time", inplace=True)
df = df.ffill().bfill()

print("正在进行工业数据去噪...")
df = NR.clean_industrial_data(df)

cols = list(df.columns)
ti = cols.index(TARGET_COL) # 获取目标列的索引
num_vars = len(cols)        # 获取总变量数

X_raw, y_raw, scaler_x, scaler_y = fit_minmax_scalers_train_only(
    df.values, df[[TARGET_COL]].values
)

X_seq, y_seq = create_sequences_multistep(X_raw, y_raw, LAG_STEPS, PREDICT_HORIZON)

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
    generator=make_dataloader_generator(),
)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=BATCH_SIZE)

# ==========================================
# 4. 训练模型
# ==========================================
raw_model = PatchTST(
    seq_len=LAG_STEPS,
    pred_len=PREDICT_HORIZON,
    num_vars=num_vars, # 必须传入总变量数用于 RevIN 和 Heads
    target_idx=ti,   
    patch_len=16,    
    stride=8,        
    d_model=128,
    n_heads=8,
    e_layers=3       # 标准的 3 层
)

if GPU_COUNT > 1:
    print(f"启用 DataParallel 进行多卡并行训练")
    model = nn.DataParallel(raw_model)
else:
    model = raw_model

model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.MSELoss() 

print(f"\n开始训练连续 {PREDICT_HORIZON} 步 标准 PatchTST...")
best_val_loss = float("inf")
patience = 7
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
        state_dict = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
        torch.save(state_dict, "best_patchtst_std_multistep.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("触发早停，结束训练")
            break

# ==========================================
# 5. 测试与指标提取
# ==========================================
checkpoint = torch.load("best_patchtst_std_multistep.pth", map_location=DEVICE)
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
        preds = model(batch_x)
        y_preds_scaled.append(preds.cpu().numpy())
        y_true_scaled.append(batch_y.numpy())

y_pred_matrix = np.concatenate(y_preds_scaled)
y_true_matrix = np.concatenate(y_true_scaled)

y_pred_inv = scaler_y.inverse_transform(y_pred_matrix.reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)
y_true_inv = scaler_y.inverse_transform(y_true_matrix.reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)

metrics = regression_metrics(y_true_inv.flatten(), y_pred_inv.flatten())
print_standard_eval_report(f"Standard PatchTST (连续 {PREDICT_HORIZON} 步)", metrics)
append_model_eval_report(
    "Standard-PatchTST-MultiStep", metrics, 
    lag_steps=LAG_STEPS, predict_horizon=PREDICT_HORIZON
)

# ==========================================
# 6. 多步连续预测的图像展示
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

pred_png = save_multistep_horizon_plot(
    y_true_inv,
    y_pred_inv,
    model_slug="patchtst",
    model_label="PatchTST",
    window_offset=val_end,
)

plt.figure(figsize=(10, 5))
plt.hist((y_true_inv - y_pred_inv).flatten(), bins=50, color="#2ca02c", alpha=0.7, edgecolor="black")
plt.title(f"Prediction Error Distribution (All {PREDICT_HORIZON} Steps - Std PatchTST)")
plt.xlabel("Error (A)")
plt.ylabel("Frequency")
plt.axvline(0, color="red", linestyle="dashed", linewidth=1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/patchtst_std_error_hist_multistep.png", dpi=120)
plt.close()

print(f"预测曲线已保存: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")

try:
    del model, optimizer, train_loader, val_loader, test_loader, criterion, checkpoint
except NameError:
    pass
release_gpu_memory()