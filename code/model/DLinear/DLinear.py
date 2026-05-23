import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Aprocess import noiseReduce as NR
from Autils.eval_config import (
    FILE_PATH, LAG_STEPS, METRICS_REPORT_PATH, OUTPUT_DIR, PLOT_TAIL,
    PREDICT_HORIZON, TARGET_COL, TRAIN_RATIO, VAL_RATIO,
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_dlinear_std_multistep.pth")

# ==========================================
# 1. 核心组件 - 严格对齐原版 DLinear 
# ==========================================
class DLinear(nn.Module):
    """
    原版 DLinear (Decomposition-Linear) 
    核心特征：移动平均分解 + 严格的通道独立 (每个变量独享权重)
    """
    def __init__(self, seq_len, pred_len, num_vars, target_idx, kernel_size=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_vars = num_vars
        self.target_idx = target_idx
        self.kernel_size = kernel_size

        # 严格的通道独立：为每个变量(通道)实例化独立的 Linear 层
        self.Linear_Seasonal = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_vars)
        ])
        self.Linear_Trend = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(num_vars)
        ])

    def forward(self, x):
        # x shape: [Batch, Seq_len, Num_vars]
        B, L, V = x.shape
        
        # 1. 简单的移动平均计算(提取 Trend)
        pad_size = (self.kernel_size - 1) // 2
        # 对时间维度的首尾进行 padding，保证滑动平均后序列长度不变
        front = x[:, 0:1, :].repeat(1, pad_size, 1)
        end = x[:, -1:, :].repeat(1, pad_size, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        
        # AvgPool1d 要求输入格式为 [Batch, Channels, Seq_len]
        trend_init = F.avg_pool1d(
            x_pad.permute(0, 2, 1), 
            kernel_size=self.kernel_size, 
            stride=1
        ).permute(0, 2, 1)
        
        # 提取残差作为季节性分量 (Seasonal)
        seasonal_init = x - trend_init

        # 2. 调整维度为 [Batch, Num_vars, Seq_len] 准备映射
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        # 初始化输出张量 [Batch, Num_vars, Pred_len]
        seasonal_output = torch.zeros([B, V, self.pred_len], dtype=x.dtype, device=x.device)
        trend_output = torch.zeros([B, V, self.pred_len], dtype=x.dtype, device=x.device)
        
        # 3. 通道独立 (Channel Independence) 的线性映射
        for i in range(self.num_vars):
            seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
            trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
            
        # 4. 组合结果并还原维度为 [Batch, Pred_len, Num_vars]
        x_out = (seasonal_output + trend_output).permute(0, 2, 1)
        
        # 5. 返回目标变量的预测 [Batch, Pred_len]
        return x_out[:, :, self.target_idx]


# ==========================================
# 2. 数据处理 (保持不变)
# ==========================================
print(f"加载数据 | 设备: {DEVICE} | GPU 数量: {GPU_COUNT}")
df = pd.read_csv(FILE_PATH, parse_dates=["time"]).sort_values("time").ffill().bfill()
df = NR.clean_industrial_data(df.set_index("time"))
cols = list(df.columns)
target_idx = cols.index(TARGET_COL)

X_raw, y_raw, scaler_x, scaler_y = fit_minmax_scalers_train_only(
    df.values, df[[TARGET_COL]].values
)
X_seq, y_seq = create_sequences_multistep(X_raw, y_raw, LAG_STEPS, PREDICT_HORIZON)

t1 = int(len(X_seq) * TRAIN_RATIO)
t2 = int(len(X_seq) * (TRAIN_RATIO + VAL_RATIO))

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_seq[:t1]), torch.FloatTensor(y_seq[:t1])),
    batch_size=128,
    shuffle=True,
    generator=make_dataloader_generator(),
)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_seq[t1:t2]), torch.FloatTensor(y_seq[t1:t2])),
    batch_size=128
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_seq[t2:]), torch.FloatTensor(y_seq[t2:])),
    batch_size=128
)

# ==========================================
# 3. 模型、优化器与训练 (保持不变)
# ==========================================
model = DLinear(LAG_STEPS, PREDICT_HORIZON, len(cols), target_idx).to(DEVICE)
if GPU_COUNT > 1:
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
criterion = nn.MSELoss()
# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

best_val = float("inf")
patience = 7
counter = 0

for epoch in range(50):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for bx, by in pbar:
        bx, by = bx.to(DEVICE), by.to(DEVICE)
        optimizer.zero_grad()
        output = model(bx)
        loss = criterion(output, by)
        loss.backward()
        # 梯度裁剪，稳定训练
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * bx.size(0)
        pbar.set_postfix({"loss": loss.item()})

    train_loss /= len(train_loader.dataset)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    total_val_samples = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            pred = model(bx)
            batch_loss = criterion(pred, by).item()
            val_loss += batch_loss * bx.size(0)
            total_val_samples += bx.size(0)
    val_loss /= total_val_samples

    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    scheduler.step(val_loss)

    # Early stopping & 保存最佳模型
    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, MODEL_SAVE_PATH)
        print(f"  -> 保存最佳模型到 {MODEL_SAVE_PATH}")
    else:
        counter += 1
        if counter >= patience:
            print(f"早停于 epoch {epoch+1}")
            break

print("训练完成。")

# ==========================================
# 4. 测试与评估 (保持不变)
# ==========================================
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)
model.eval()

preds, trues = [], []
with torch.no_grad():
    for bx, by in test_loader:
        pred = model(bx.to(DEVICE)).cpu().numpy()
        preds.append(pred)
        trues.append(by.numpy())

y_p = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)
y_t = scaler_y.inverse_transform(np.concatenate(trues).reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)

metrics = regression_metrics(y_t.flatten(), y_p.flatten())
print_standard_eval_report("Standard DLinear ", metrics)

append_model_eval_report(
    "DLinear-Std",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
pred_png = save_multistep_horizon_plot(
    y_t,
    y_p,
    model_slug="dlinear_fixed",
    model_label="DLinear-Fixed",
    window_offset=t2,
)
print(f"预测曲线保存至: {pred_png}")

release_gpu_memory()