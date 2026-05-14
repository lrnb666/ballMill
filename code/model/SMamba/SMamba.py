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
# 1. 配置（见 Autils/eval_config；PRED_STEPS 与 PREDICT_HORIZON 等价）
# ==========================================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 数据
# ==========================================
print(f"加载数据 | 设备: {DEVICE}")
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
# 3. S-Mamba
# ==========================================
class SMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_forward = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_backward = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x_fwd = self.mamba_forward(x)
        x_bwd = self.mamba_backward(x.flip(dims=[1]))
        x_bwd = x_bwd.flip(dims=[1])
        x = self.norm1(x_fwd + x_bwd + residual)
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        return x


class SMamba(nn.Module):
    def __init__(self, seq_len, n_features, d_model=128, n_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.token_linear = nn.Linear(seq_len, d_model)
        self.layers = nn.ModuleList([SMambaBlock(d_model) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        B, L, V = x.shape
        x = x.transpose(1, 2)
        x = self.token_linear(x)
        for layer in self.layers:
            x = layer(x)
        x = self.proj(x)
        x = x.mean(dim=1)
        return x


model = SMamba(seq_len=LAG_STEPS, n_features=X_raw.shape[1]).to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"DataParallel | GPU={torch.cuda.device_count()}")
    model = nn.DataParallel(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.MSELoss()

# ==========================================
# 4. 训练
# ==========================================
print("\n开始训练 S-Mamba...")
best_val_loss = float("inf")
patience = 10
counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    for batch_x, batch_y in train_pbar:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.6f}")

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            total_val_loss += criterion(model(batch_x), batch_y).item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"--- Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(sd, "best_smamba_model.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("早停")
            break

# ==========================================
# 5. 评估
# ==========================================
state_dict = torch.load("best_smamba_model.pth", map_location=DEVICE)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)
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
print_standard_eval_report("S-Mamba", metrics)
append_model_eval_report(
    "S-Mamba",
    metrics,
    lag_steps=LAG_STEPS,
    predict_horizon=PREDICT_HORIZON,
)

pred_png = save_prediction_plot(
    y_true,
    y_pred,
    out_filename="smamba_predict.png",
    title=f"Ball Mill Current — S-Mamba (H={PREDICT_HORIZON}, last {PLOT_TAIL})",
    pred_label="S-Mamba Pred",
)
print(f"预测曲线: {pred_png}")
print(f"指标已追加: {METRICS_REPORT_PATH}")

if metrics["industrial_hit_pct"] > 80:
    print("评估结论：工业命中率 >80%。")
else:
    print("评估结论：可继续调 LAG_STEPS / d_model。")


def predict_next_horizon(recent_data_df):
    """最近 LAG_STEPS 行（与训练列一致）→ 未来第 PREDICT_HORIZON 步电流。"""
    core = model.module if isinstance(model, nn.DataParallel) else model
    core.eval()
    with torch.no_grad():
        input_scaled = scaler_x.transform(recent_data_df.values)
        input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(DEVICE)
        pred_scaled = core(input_tensor)
        return float(scaler_y.inverse_transform(pred_scaled.cpu().numpy())[0, 0])


predict_next_minute = predict_next_horizon
