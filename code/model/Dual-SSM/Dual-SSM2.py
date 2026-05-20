"""
Dual-SSM (Dual-Stream Spectral and State-Space Model) — DDP双卡 Gated改进版
------------------------------------
核心架构设计：
1. 保持 TimesNet 的 InceptionBlock 不变，提取频域多周期特征。
2. 引入 Mamba 扫描路径提取时域状态特征，增加前后线性投影防止维度错位。
3. [核心改进] 引入 Gated Fusion (门控融合)，废弃固定的 alpha，动态计算双流特征权重。
4. [核心改进] 增加双流独立 LayerNorm，防止梯度劫持和特征互相覆盖。
5. 优化训练循环，为门控网络赋予更高的独立学习率，加速双流权重分配。
6. 支持 python xxx.py 傻瓜式一键双卡/多卡 DDP 训练。
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.fft
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from mamba_ssm import Mamba

# 注意：请确保你的本地环境存在这三个包/文件
from Aprocess import noiseReduce as NR
from Autils.eval_config import (
    FILE_PATH, LAG_STEPS, PREDICT_HORIZON, TARGET_COL, TRAIN_RATIO, VAL_RATIO,
    append_model_eval_report, create_sequences_multistep,
    fit_minmax_scalers_train_only, industrial_hit_rate_pct,
    make_dataloader_generator, print_standard_eval_report,
    regression_metrics, release_gpu_memory,
    save_multistep_horizon_plot, set_global_seed,
    INDUSTRIAL_HIT_REL_TOLERANCE, OUTPUT_DIR
)

warnings.filterwarnings("ignore")
set_global_seed()

# 将模型保存在 OUTPUT_DIR 下防止相对路径找不到
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_Dual_SSM_ddp.pth")

# ==========================================
# 1. 核心组件 
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        
    def forward(self, x, mode, stats=None):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return ((x - self.mean) / self.stdev) * self.gamma + self.beta, (self.mean, self.stdev)
        elif mode == 'denorm':
            mean, stdev = stats  
            return ((x - self.beta) / self.gamma) * stdev + mean

class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.project = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.gelu = nn.GELU()
        
        # 修复极小通道 GroupNorm 报错问题
        self.norm = nn.GroupNorm(min(4, out_channels), out_channels) if out_channels >= 2 else nn.Identity()

    def forward(self, x):
        x_res = x.contiguous()
        x_cat = torch.cat([self.conv1(x_res), self.conv3(x_res), self.conv5(x_res)], dim=1)
        out = self.gelu(self.norm(self.project(x_cat)))
        return out + x_res

# ==========================================
# 2. Dual-SSM 核心块 (Gated Fusion 改进版)
# ==========================================
class DualSSMBlock(nn.Module):
    def __init__(self, d_model, top_k, dropout=0.1):
        super().__init__()
        self.k = top_k
        self.d_model = d_model
        
        # [修复] 双流独立归一化，防止特征互相干扰
        self.norm_spectral = nn.LayerNorm(d_model)
        self.norm_state = nn.LayerNorm(d_model)
        
        # --- 频域分支 (Spectral Stream) ---
        self.inception = InceptionBlockV1(d_model, d_model)
        
        # --- 时域分支 (State-Space Stream) ---
        # [修复] 增加前后线性投影，保护 Mamba 状态空间
        self.mamba_proj_in = nn.Linear(d_model, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_proj_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # [修复] 门控融合机制 (Gated Fusion) 替代固定 alpha
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 运行时变量：记录当前 batch 平均 Mamba 占比（方便终端打印）
        self.current_mamba_ratio = 0.5

    def forward(self, x):
        B, T, C = x.shape
        
        # ==========================================
        # Stream 1: State-Space (Mamba) 独立路径
        # ==========================================
        x_state = self.norm_state(x)
        x_state = self.mamba_proj_in(x_state)
        mamba_out = self.mamba(x_state)
        mamba_out = self.mamba_proj_out(mamba_out)
        
        # ==========================================
        # Stream 2: Spectral (FFT + Inception) 独立路径
        # ==========================================
        x_spectral = self.norm_spectral(x)
        xf = torch.fft.rfft(x_spectral, dim=1)
        amp = torch.abs(xf).mean(dim=0).mean(dim=-1)
        amp[0] = 0  
        
        valid_freqs = (amp > 0).sum().item()
        k = min(self.k, valid_freqs)
        
        inception_res = torch.zeros_like(x_spectral)
        if k > 0:
            top_amps, top_freqs = torch.topk(amp, k)
            weights = torch.softmax(top_amps, dim=0)
            
            for i in range(k):
                freq = top_freqs[i].item()
                period = max(T // freq, 1) if freq > 0 else T
                period = min(period, T)
                
                pad_len = (period - T % period) % period
                if pad_len > 0:
                    x_pad = torch.cat([x_spectral, x_spectral[:, -1:, :].repeat(1, pad_len, 1)], dim=1)
                else:
                    x_pad = x_spectral
                T_pad = x_pad.shape[1]
                
                x_2d = x_pad.transpose(1, 2).contiguous().reshape(B, C, T_pad // period, period)
                out_2d = self.inception(x_2d)
                out_1d = out_2d.reshape(B, C, T_pad).transpose(1, 2).contiguous()
                inception_res += out_1d[:, :T, :] * weights[i]

        # ==========================================
        # 深度门控融合 (Gated Fusion)
        # ==========================================
        concat_features = torch.cat([mamba_out, inception_res], dim=-1)
        g = self.fusion_gate(concat_features) 
        
        # 记录均值，供控制台打印监控 (g越接近1代表Mamba权重越大)
        self.current_mamba_ratio = g.mean().item()
        
        # 动态按比例融合
        fused = g * mamba_out + (1 - g) * inception_res 
        fused = self.out_proj(fused)

        return x + self.dropout(fused)

class DualSSM(nn.Module):
    def __init__(self, seq_len, pred_len, num_vars, target_idx, d_model=64, e_layers=2, top_k=2):
        super().__init__()
        self.seq_len, self.pred_len, self.num_vars, self.target_idx = seq_len, pred_len, num_vars, target_idx
        self.revin = RevIN(num_vars)
        self.enc_embedding = nn.Linear(num_vars, d_model)
        
        # 堆叠 DualSSM Block
        self.blocks = nn.ModuleList([DualSSMBlock(d_model, top_k) for _ in range(e_layers)])
        
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(seq_len * d_model, pred_len * num_vars)
        )

    def forward(self, x):
        B, L, V = x.shape
        x, stats = self.revin(x, mode='norm') 
        x_enc = self.enc_embedding(x)
        for block in self.blocks: 
            x_enc = block(x_enc)
        out = self.head(x_enc).reshape(B, self.pred_len, self.num_vars)
        return self.revin(out, mode='denorm', stats=stats)[:, :, self.target_idx]

# ==========================================
# 3. 内部 Worker (负责每张卡上的训练)
# ==========================================
def train_worker(local_rank, world_size):
    # 1. 初始化进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    if local_rank == 0:
        print(f"Dual-SSM | 启动双卡 DDP 分布式训练 (可用显卡: {world_size})...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. 数据准备
    df = pd.read_csv(FILE_PATH, parse_dates=["time"]).sort_values("time").ffill().bfill()
    df = NR.clean_industrial_data(df.set_index("time"))
    cols = list(df.columns)
    target_idx = cols.index(TARGET_COL)
    
    X_raw, y_raw, _, scaler_y = fit_minmax_scalers_train_only(df.values, df[[TARGET_COL]].values)
    X_seq, y_seq = create_sequences_multistep(X_raw, y_raw, LAG_STEPS, PREDICT_HORIZON)

    t1, t2 = int(len(X_seq)*TRAIN_RATIO), int(len(X_seq)*(TRAIN_RATIO+VAL_RATIO))
    
    train_dataset = TensorDataset(torch.FloatTensor(X_seq[:t1]), torch.FloatTensor(y_seq[:t1]))
    val_dataset = TensorDataset(torch.FloatTensor(X_seq[t1:t2]), torch.FloatTensor(y_seq[t1:t2]))
    test_dataset = TensorDataset(torch.FloatTensor(X_seq[t2:]), torch.FloatTensor(y_seq[t2:]))

    # 3. DDP 数据切分 Sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    PER_GPU_BATCH = 128 // world_size
    train_loader = DataLoader(train_dataset, batch_size=PER_GPU_BATCH, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=PER_GPU_BATCH, sampler=val_sampler, num_workers=4, pin_memory=True)

    # 4. 模型与优化器设定
    model = DualSSM(LAG_STEPS, PREDICT_HORIZON, len(cols), target_idx, d_model=64).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # [修复] 为 fusion_gate 分配 10倍的学习率，让其快速学会平衡双流
    base_params = [p for n, p in model.named_parameters() if 'fusion_gate' not in n]
    gate_params = [p for n, p in model.named_parameters() if 'fusion_gate' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 0.001},
        {'params': gate_params, 'lr': 0.01}  
    ], weight_decay=1e-4)
    
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=(local_rank==0))

    # 5. 训练主循环
    best_val, patience, counter = float("inf"), 8, 0
    for epoch in range(50):
        train_sampler.set_epoch(epoch) 
        model.train()
        
        train_loss_sum = 0
        train_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}") if local_rank == 0 else train_loader
        
        for bx, by in pbar:
            bx, by = bx.to(local_rank), by.to(local_rank)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_sum += loss.item() * bx.size(0)
            train_samples += bx.size(0)
        
        # 跨卡同步计算 Train Loss
        train_tensor = torch.tensor([train_loss_sum, train_samples], device=local_rank)
        dist.all_reduce(train_tensor, op=dist.ReduceOp.SUM)
        global_train_loss = (train_tensor[0] / train_tensor[1]).item()
        
        # --- 验证集评估 ---
        model.eval()
        val_loss_sum = 0
        val_samples = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(local_rank), by.to(local_rank)
                batch_loss = criterion(model(bx), by).item()
                val_loss_sum += batch_loss * bx.size(0)
                val_samples += bx.size(0)
        
        # 跨卡同步计算 Val Loss
        val_tensor = torch.tensor([val_loss_sum, val_samples], device=local_rank)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        global_val_loss = (val_tensor[0] / val_tensor[1]).item()
        
        # [修改] 打印 Loss 和 每一层当前的 Mamba 融合占比百分比
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            gate_infos = []
            for i, block in enumerate(model.module.blocks):
                # 乘 100 转换为百分比
                mamba_pct = block.current_mamba_ratio * 100 
                gate_infos.append(f"L{i}_Mamba:{mamba_pct:.1f}%")
            
            gate_str = " | ".join(gate_infos)
            print(f"Epoch {epoch+1} | Train Loss: {global_train_loss:.6f} | Val Loss: {global_val_loss:.6f} | LR: {current_lr:.6f} | Weights: [{gate_str}]")
        
        scheduler.step(global_val_loss)
        
        # 早停判断
        if global_val_loss < best_val:
            best_val = global_val_loss
            counter = 0
            if local_rank == 0:
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH) 
        else:
            counter += 1
            if counter >= patience: 
                if local_rank == 0:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    dist.barrier()

    # 6. 单卡测试与评估 
    if local_rank == 0:
        print("\n[Rank 0] 加载最佳权重进行测试评估...")
        test_model = DualSSM(LAG_STEPS, PREDICT_HORIZON, len(cols), target_idx, d_model=64).to(local_rank)
        test_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=f'cuda:{local_rank}'))
        test_model.eval()
        
        test_loader_single = DataLoader(test_dataset, batch_size=128, shuffle=False)
        preds, trues = [], []
        with torch.no_grad():
            for bx, by in test_loader_single:
                preds.append(test_model(bx.to(local_rank)).cpu().numpy())
                trues.append(by.numpy())

        y_p = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)
        y_t = scaler_y.inverse_transform(np.concatenate(trues).reshape(-1, 1)).reshape(-1, PREDICT_HORIZON)

        metrics = regression_metrics(y_t.flatten(), y_p.flatten())
        print_standard_eval_report(f"Dual-SSM (连续 {PREDICT_HORIZON} 步)", metrics)
        append_model_eval_report(
            "Dual-SSM-MultiStep", metrics, 
            lag_steps=LAG_STEPS, predict_horizon=PREDICT_HORIZON
        )

        save_multistep_horizon_plot(y_t, y_p, model_slug="dual_ssm_ddp", model_label="Dual-SSM", window_offset=t2)
        release_gpu_memory()

    dist.destroy_process_group()

# ==========================================
# 4. 外部启动器
# ==========================================
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("未检测到多张显卡，自动回退到单卡训练。")
        train_worker(0, 1)