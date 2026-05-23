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

# 你的本地依赖包
from Aprocess import noiseReduce as NR
from Autils.eval_config import (
    FILE_PATH, TARGET_COL, TRAIN_RATIO, VAL_RATIO,
    append_model_eval_report, create_sequences_multistep,
    fit_minmax_scalers_train_only, print_standard_eval_report,
    regression_metrics, release_gpu_memory,
    save_multistep_horizon_plot, set_global_seed,
    OUTPUT_DIR
)

warnings.filterwarnings("ignore")
set_global_seed()

# ==========================================
# 1. 核心组件 (保持不变)
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
        self.norm = nn.GroupNorm(min(4, out_channels), out_channels) if out_channels >= 2 else nn.Identity()

    def forward(self, x):
        x_res = x.contiguous()
        x_cat = torch.cat([self.conv1(x_res), self.conv3(x_res), self.conv5(x_res)], dim=1)
        out = self.gelu(self.norm(self.project(x_cat)))
        return out + x_res

class DualSSMBlock(nn.Module):
    def __init__(self, d_model, top_k, dropout=0.1):
        super().__init__()
        self.k = top_k
        self.d_model = d_model
        
        self.norm_spectral = nn.LayerNorm(d_model)
        self.norm_state = nn.LayerNorm(d_model)
        self.inception = InceptionBlockV1(d_model, d_model)
        self.mamba_proj_in = nn.Linear(d_model, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_proj_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_history = []

    def forward(self, x):
        B, T, C = x.shape
        x_state = self.norm_state(x)
        mamba_in = self.mamba_proj_in(x_state)
        mamba_out = self.mamba_proj_out(self.mamba(mamba_in)) + x_state
        
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
                x_pad = torch.cat([x_spectral, x_spectral[:, -1:, :].repeat(1, pad_len, 1)], dim=1) if pad_len > 0 else x_spectral
                T_pad = x_pad.shape[1]
                x_2d = x_pad.transpose(1, 2).contiguous().reshape(B, C, T_pad // period, period)
                out_2d = self.inception(x_2d)
                out_1d = out_2d.reshape(B, C, T_pad).transpose(1, 2).contiguous()
                inception_res += out_1d[:, :T, :] * weights[i]

        concat_features = torch.cat([mamba_out, inception_res], dim=-1)
        g = self.fusion_gate(concat_features) 
        if not self.training:
            self.gate_history.append(g.mean().item())
        fused = g * mamba_out + (1 - g) * inception_res 
        fused = self.out_proj(fused)
        return x + self.dropout(fused)

class DualSSM(nn.Module):
    def __init__(self, seq_len, pred_len, num_vars, target_idx, d_model=64, e_layers=2, top_k=2):
        super().__init__()
        self.seq_len, self.pred_len, self.num_vars, self.target_idx = seq_len, pred_len, num_vars, target_idx
        self.revin = RevIN(num_vars)
        self.enc_embedding = nn.Linear(num_vars, d_model)
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
# [修改] 增加 lag_steps 和 pred_steps 作为函数参数
def train_worker(local_rank, world_size, lag_steps, pred_steps):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    
    # [修改] 动态设置当前实验的模型保存路径
    model_save_path = os.path.join(OUTPUT_DIR, f"best_Dual_SSM_in{lag_steps}_out{pred_steps}.pth")
    
    if local_rank == 0:
        print(f"\nDual-SSM | DDP 训练启动 | 输入: {lag_steps} -> 预测: {pred_steps}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(FILE_PATH, parse_dates=["time"]).sort_values("time").ffill().bfill()
    df = NR.clean_industrial_data(df.set_index("time"))
    cols = list(df.columns)
    target_idx = cols.index(TARGET_COL)
    
    X_raw, y_raw, _, scaler_y = fit_minmax_scalers_train_only(df.values, df[[TARGET_COL]].values)
    # [修改] 使用传入的动态参数构建序列
    X_seq, y_seq = create_sequences_multistep(X_raw, y_raw, lag_steps, pred_steps)

    t1, t2 = int(len(X_seq)*TRAIN_RATIO), int(len(X_seq)*(TRAIN_RATIO+VAL_RATIO))
    
    train_dataset = TensorDataset(torch.FloatTensor(X_seq[:t1]), torch.FloatTensor(y_seq[:t1]))
    val_dataset = TensorDataset(torch.FloatTensor(X_seq[t1:t2]), torch.FloatTensor(y_seq[t1:t2]))
    test_dataset = TensorDataset(torch.FloatTensor(X_seq[t2:]), torch.FloatTensor(y_seq[t2:]))

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    PER_GPU_BATCH = 160 // world_size
    train_loader = DataLoader(train_dataset, batch_size=PER_GPU_BATCH, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=PER_GPU_BATCH, sampler=val_sampler, num_workers=4, pin_memory=True)

    # [修改] 使用传入的动态参数初始化模型
    model = DualSSM(lag_steps, pred_steps, len(cols), target_idx, d_model=64).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    base_params = [p for n, p in model.named_parameters() if 'fusion_gate' not in n]
    gate_params = [p for n, p in model.named_parameters() if 'fusion_gate' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 0.001},
        {'params': gate_params, 'lr': 0.01}  
    ], weight_decay=1e-4)
    
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=(local_rank==0))

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
        
        val_tensor = torch.tensor([val_loss_sum, val_samples], device=local_rank)
        dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
        global_val_loss = (val_tensor[0] / val_tensor[1]).item()
        
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            gate_infos = []
            for i, block in enumerate(model.module.blocks):
                avg_mamba_ratio = sum(block.gate_history) / len(block.gate_history) if len(block.gate_history) > 0 else 0.5
                mamba_pct = avg_mamba_ratio * 100 
                fft_pct = 100 - mamba_pct
                gate_infos.append(f"L{i}(Mamba:{mamba_pct:.1f}%|FFT:{fft_pct:.1f}%)")
            
            gate_str = " - ".join(gate_infos)
            print(f"Epoch {epoch+1} | Train Loss: {global_train_loss:.5f} | Val Loss: {global_val_loss:.5f} | LR: {current_lr:.6f} | {gate_str}")
        
        for block in model.module.blocks:
            block.gate_history.clear()
            
        scheduler.step(global_val_loss)
        
        if global_val_loss < best_val:
            best_val = global_val_loss
            counter = 0
            if local_rank == 0:
                torch.save(model.module.state_dict(), model_save_path) 
        else:
            counter += 1
            if counter >= patience: 
                if local_rank == 0:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    dist.barrier()

    # ==========================================
    # 测试与【逐步(Step-wise)】评估阶段
    # ==========================================
    if local_rank == 0:
        print(f"\n[Rank 0] 加载最佳权重进行测试评估 ({lag_steps} 步预测 {pred_steps} 步)...")
        test_model = DualSSM(lag_steps, pred_steps, len(cols), target_idx, d_model=64).to(local_rank)
        test_model.load_state_dict(torch.load(model_save_path, map_location=f'cuda:{local_rank}'))
        test_model.eval()
        
        test_loader_single = DataLoader(test_dataset, batch_size=128, shuffle=False)
        preds, trues = [], []
        with torch.no_grad():
            for bx, by in test_loader_single:
                preds.append(test_model(bx.to(local_rank)).cpu().numpy())
                trues.append(by.numpy())

        # 形状恢复为 (Samples, pred_steps)
        y_p = scaler_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).reshape(-1, pred_steps)
        y_t = scaler_y.inverse_transform(np.concatenate(trues).reshape(-1, 1)).reshape(-1, pred_steps)

        # 1. 整体均值评估 (保持你原有的输出格式)
        metrics_global = regression_metrics(y_t.flatten(), y_p.flatten())
        print_standard_eval_report(f"Dual-SSM 全局指标 ({pred_steps} 步)", metrics_global)
        append_model_eval_report(
            f"Dual-SSM-MultiStep-In{lag_steps}-Out{pred_steps}", metrics_global, 
            lag_steps=lag_steps, predict_horizon=pred_steps
        )
        save_multistep_horizon_plot(y_t, y_p, model_slug=f"dual_ssm_in{lag_steps}_out{pred_steps}", model_label="Dual-SSM", window_offset=t2)

        # -----------------------------------------------------
        # 2. [新增] 逐步(每一分钟)准确率统计与保存 
        # -----------------------------------------------------
        step_metrics_list = []
        for step in range(pred_steps):
            yt_step = y_t[:, step]
            yp_step = y_p[:, step]
            
            # 手动计算各项核心回归指标
            mse = np.mean((yt_step - yp_step)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(yt_step - yp_step))
            
            # 防止除以 0 计算 MAPE
            mask = yt_step != 0
            mape = np.mean(np.abs((yt_step[mask] - yp_step[mask]) / yt_step[mask])) * 100 if np.any(mask) else 0.0
            
            # 你可以在此处加入你自己定义的 industrial_hit_rate_pct 逻辑
            # hit_rate = industrial_hit_rate_pct(yt_step, yp_step, ...) 
            
            step_metrics_list.append({
                "Future_Step": step + 1,  # 第1分钟, 第2分钟 ...
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "MAPE(%)": round(mape, 4)
            })
            
        # 将逐步结果转换为 DataFrame 并保存为单独的 CSV
        df_step_metrics = pd.DataFrame(step_metrics_list)
        step_csv_filename = os.path.join(OUTPUT_DIR, f"DualSSM_Stepwise_Acc_in{lag_steps}_out{pred_steps}.csv")
        df_step_metrics.to_csv(step_csv_filename, index=False)
        
        print("\n" + "="*50)
        print(f"✅ 逐步预测准确率已保存至: {step_csv_filename}")
        print("前 5 步准确率预览:")
        print(df_step_metrics.head(5).to_string(index=False))
        print("="*50 + "\n")

        release_gpu_memory()

    dist.destroy_process_group()

# ==========================================
# 4. 外部启动器 (支持多实验批量运行)
# ==========================================
if __name__ == "__main__":
    
    # [新增] 在这里配置你想跑的实验列表
    # 格式：{"lag": 过去多少分钟, "pred": 预测未来多少分钟}
    experiments = [
       # {"lag": 120, "pred": 20},
     #   {"lag": 120, "pred": 30},
         {"lag": 120, "pred": 60}, # 可以随时解除注释，让它排队自己跑
    ]
    
    world_size = torch.cuda.device_count()
    
    for exp in experiments:
        lag = exp["lag"]
        pred = exp["pred"]
        
        print(f"\n{'#'*60}")
        print(f" 即将启动实验: 使用过去 {lag} 分钟，预测未来 {pred} 分钟")
        print(f"{'#'*60}")
        
        if world_size > 1:
            # 传入多出的参数: args=(world_size, lag, pred)
            mp.spawn(train_worker, args=(world_size, lag, pred), nprocs=world_size, join=True)
        else:
            print("未检测到多张显卡，自动回退到单卡训练。")
            train_worker(0, 1, lag, pred)
        
        print(f"✅ 实验 (In:{lag} -> Out:{pred}) 运行结束！\n")