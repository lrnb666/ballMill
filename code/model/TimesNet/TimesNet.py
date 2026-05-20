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
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_timesnet_std_multistep.pth")

# ==========================================
# 1. 核心组件（修正版）
# ==========================================
class RevIN(nn.Module):
    """可逆实例归一化，兼容 DataParallel
       改进：forward 返回统计量，denorm 时显式传入，避免多卡统计量不一致
    """
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode, stats=None):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            normalized = ((x - self.mean) / self.stdev) * self.gamma + self.beta
            # 返回归一化后的数据和当前 batch 的统计量，供 denorm 使用
            return normalized, (self.mean, self.stdev)
        elif mode == 'denorm':
            if stats is None:
                raise ValueError("RevIN denorm 需要传入 stats 参数 (mean, stdev)")
            mean, stdev = stats
            return ((x - self.beta) / self.gamma) * stdev + mean


class InceptionBlockV1(nn.Module):
    """Inception 块，增加残差连接"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.project = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.gelu = nn.GELU()
        # 当 out_channels 小于 4 时，GroupNorm 的组数不能超过通道数
        self.norm = nn.GroupNorm(min(4, out_channels), out_channels) if out_channels >= 2 else nn.Identity()

    def forward(self, x):
        identity = x.contiguous()
        x_cat = torch.cat([self.conv1(identity), self.conv3(identity), self.conv5(identity)], dim=1)
        out = self.gelu(self.norm(self.project(x_cat)))
        # 残差连接，要求 in_channels == out_channels（满足）
        return out + identity


class TimesBlock(nn.Module):
    def __init__(self, d_model, top_k):
        super().__init__()
        self.k = top_k
        self.norm1 = nn.LayerNorm(d_model)
        self.inception = InceptionBlockV1(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        x_norm = self.norm1(x)
        xf = torch.fft.rfft(x_norm, dim=1)
        amp = torch.abs(xf).mean(dim=0).mean(dim=-1)
        amp[0] = 0
        k = min(self.k, len(amp))
        top_amps, top_freqs = torch.topk(amp, k)
        weights = torch.softmax(top_amps, dim=0)

        res = torch.zeros_like(x_norm)
        for i in range(k):
            freq = top_freqs[i].item()
            period = max(T // freq, 1) if freq > 0 else T
            # 确保周期不小于卷积核大小，避免 2D 卷积在过窄维度上失效
            period = max(period, 5)
            pad_len = (period - T % period) % period

            if pad_len > 0:
                x_pad = torch.cat([x_norm, x_norm[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
            else:
                x_pad = x_norm

            T_pad = x_pad.shape[1]
            # 强制连续内存防止内存对齐错误
            x_2d = x_pad.transpose(1, 2).contiguous().reshape(B, C, T_pad // period, period)
            out_2d = self.inception(x_2d)
            out_1d = out_2d.reshape(B, C, T_pad).transpose(1, 2).contiguous()
            res += out_1d[:, :T, :] * weights[i]

        return x + self.dropout(res)


class TimesNet(nn.Module):
    def __init__(self, seq_len, pred_len, num_vars, target_idx, d_model=64, e_layers=2, top_k=3):
        super().__init__()
        self.seq_len, self.pred_len, self.num_vars, self.target_idx = seq_len, pred_len, num_vars, target_idx
        self.revin = RevIN(num_vars)
        self.enc_embedding = nn.Linear(num_vars, d_model)
        self.blocks = nn.ModuleList([TimesBlock(d_model, top_k) for _ in range(e_layers)])
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(seq_len * d_model, pred_len * num_vars)
        )

    def forward(self, x):
        B, L, V = x.shape
        # RevIN norm 阶段返回归一化数据和统计量
        x, stats = self.revin(x, mode='norm')
        x_enc = self.enc_embedding(x)
        for block in self.blocks:
            x_enc = block(x_enc)
        out = self.head(x_enc).reshape(B, self.pred_len, self.num_vars)
        # denorm 时传入原 batch 对应的统计量
        return self.revin(out, mode='denorm', stats=stats)[:, :, self.target_idx]


# ==========================================
# 2. 数据处理
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
# 3. 模型、优化器与训练
# ==========================================
model = TimesNet(LAG_STEPS, PREDICT_HORIZON, len(cols), target_idx).to(DEVICE)
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

    # 验证阶段（加权平均损失）
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
        # 处理 DataParallel 包装
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
# 4. 测试与评估
# ==========================================
checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
# 加载权重前确保模型架构一致（处理 DataParallel）
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)
model.eval()

preds, trues = [], []
with torch.no_grad():
    for bx, by in test_loader:
        # DataParallel 下模型调用会自动拆分 batch，forward 中 RevIN 会正确处理每个子 batch
        pred = model(bx.to(DEVICE)).cpu().numpy()
        preds.append(pred)
        trues.append(by.numpy())

y_p = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)
y_t = scaler_y.inverse_transform(np.concatenate(trues).reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)

metrics = regression_metrics(y_t.flatten(), y_p.flatten())
print_standard_eval_report("Standard TimesNet (修复版)", metrics)

append_model_eval_report(
    "TimesNet-Std",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
pred_png = save_multistep_horizon_plot(
    y_t,
    y_p,
    model_slug="timesnet_fixed",
    model_label="TimesNet-Fixed",
    window_offset=t2,
)
print(f"预测曲线保存至: {pred_png}")

release_gpu_memory()