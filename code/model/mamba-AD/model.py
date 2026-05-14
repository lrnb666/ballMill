import torch
import torch.nn as nn
from mamba_ssm import Mamba

# 1. RevIN: 解决 350A 基准偏移的核心
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.eps = eps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.affine_weight = None

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,彰显=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine_weight is not None:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine_weight is not None:
                x = (x - self.affine_bias) / self.affine_weight
            x = x * self.stdev + self.mean
        return x

# 2. Series Decomposition: 提取 350A 趋势
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [Batch, Seq, Channel] -> [Batch, Channel, Seq]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend

# 3. Bidirectional Mamba Block:
class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        # x: [B, L, D]
        x_fwd = self.mamba_fwd(x)
        x_bwd = torch.flip(self.mamba_bwd(torch.flip(x, [1])), [1])
        
        out = self.norm(x_fwd + x_bwd)
        out = out + self.ffn(out) # 残差连接
        return out

class MambaAD(nn.Module):
    def __init__(self, c_in, seq_len, d_model=64, kernel_size=25):
        super().__init__()
        self.revin = RevIN(c_in)
        self.decomp = SeriesDecomp(kernel_size)
        
        # 细节流 (Seasonal Stream)
        self.enc_embedding = nn.Linear(c_in, d_model)
        self.mamba_layers = nn.ModuleList([BiMambaBlock(d_model) for _ in range(3)])
        
        # 趋势流 (Trend Stream)
        self.trend_model = nn.Linear(c_in, d_model)
        
        # 融合与输出
        self.projection = nn.Linear(d_model, c_in)

    def forward(self, x):
        # 1. RevIN 归一化
        x = self.revin(x, mode='norm')
        
        # 2. 分解
        seasonal, trend = self.decomp(x)
        
        # 3. 细节流处理 (Mamba)
        s_emb = self.enc_embedding(seasonal)
        for layer in self.mamba_layers:
            s_emb = layer(s_emb)
        s_out = self.projection(s_emb)
        
        # 4. 趋势流处理 (MLP)
        t_out = self.projection(self.trend_model(trend))
        
        # 5. 融合并反归一化
        out = s_out + t_out
        out = self.revin(out, mode='denorm')
        return out