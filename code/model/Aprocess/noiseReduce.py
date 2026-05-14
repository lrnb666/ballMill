import numpy as np
import pandas as pd

def hampel_filter(df, col, window_size=7, n_sigmas=3):
    """
    改进版 Hampel：严格使用 center=False，并建议使用更稳健的 MAD
    """
    # 必须使用 center=False，否则会引入未来数据
    rolling_median = df[col].rolling(window=window_size, center=False).median()
    
    # 使用 MAD (中位数绝对偏差) 代替标准差，对异常值更鲁棒
    def get_mad(x):
        return np.median(np.abs(x - np.median(x)))
    
    rolling_mad = df[col].rolling(window=window_size, center=False).apply(get_mad, raw=True)
    
    # 1.4826 是正态分布下 MAD 与标准差的转换系数
    outliers = (np.abs(df[col] - rolling_median) > (n_sigmas * 1.4826 * rolling_mad))
    
    # 修正：避免直接修改原始 df 可能带来的 SettingWithCopyWarning
    new_col = df[col].copy()
    new_col[outliers] = rolling_median[outliers]
    df[col] = new_col
    return df

def clean_industrial_data(df):
    """
    针对实时预测优化的工业去噪逻辑
    """
    # 1. 分类列
    temp_cols = [c for c in df.columns if any(k in c for k in ['Temp', 'Bearing', 'IR'])]
    vibrate_cols = [c for c in df.columns if 'Vibrate' in c]

    # 2. 处理温度：使用左对齐窗口均值 (center=False)
    for col in temp_cols:
        # min_periods=1 保证开头数据不丢失，但要意识到开头几个点的波动会大
        df[col] = df[col].rolling(window=10, min_periods=1, center=False).mean()

    # 3. 处理振动：禁止使用全局 quantile (防止泄露)
    # 建议使用滑动窗口的分位数，模拟实时生产中的局部异常判定
    for col in vibrate_cols:
        # 比如取过去 60 分钟的 99.5% 分位数作为上限
        limit_upper = df[col].rolling(window=60, min_periods=1).quantile(0.995)
        limit_lower = df[col].rolling(window=60, min_periods=1).quantile(0.005)
        df[col] = df[col].clip(lower=limit_lower, upper=limit_upper, axis=0)

    # 4. 处理电流：Hampel + 指数加权移动平均 (EWMA)
    # 为什么放弃 Savgol？因为 Savgol 在不看未来的情况下（单向拟合）效果很差，且容易产生阶跃滞后。
    # 工业上更常用 EWMA，它天然是因果性的（Causal）。
    
    # 先用 Hampel 去除传感器随机跳变 (Spikes)
    df = hampel_filter(df, 'Current_A', window_size=7, n_sigmas=3)
    
    # 使用指数加权平均平滑，span=5 约等于 5 分钟平滑，既平滑又灵敏
    df['Current_A'] = df['Current_A'].ewm(span=5, adjust=False).mean()
    
    return df