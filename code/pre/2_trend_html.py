import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

# ========== 1. 配置路径 (保持 CONFIG 风格) ==========
CONFIG = {
    'main_dir': r'E:\ballMill',
    'output_dir': r'E:\ballMill\output',
    'fig_height_per_row': 280,  # 每行子图分配的高度
}

# ========== 2. 读取与合并数据 ==========
search_pattern = os.path.join(CONFIG['output_dir'], 'processed_1min_data_*.csv')
file_list = glob.glob(search_pattern)

if not file_list:
    print(f"❌ 找不到预处理文件！")
    exit()

df_list = [pd.read_csv(f, index_col=0, parse_dates=True) for f in file_list]
all_data = pd.concat(df_list)
all_data = all_data[~all_data.index.duplicated(keep='last')].sort_index()

# ========== 3. 生成图表 (参考分析系统的布局样式) ==========
num_rows = len(all_data.columns)
total_height = max(850, num_rows * CONFIG['fig_height_per_row'])

fig = make_subplots(
    rows=num_rows, 
    cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.04,  # 适当的间距，防止标题重叠
    subplot_titles=[f"<b>{col}</b> 趋势分析" for col in all_data.columns]
)

# 工业级配色方案 (循环使用)
colors = ['#2E86C1', '#F39C12', '#C0392B', '#27AE60', '#8E44AD', '#16A085']

for i, col in enumerate(all_data.columns):
    # 数据平滑处理 (窗口设为5)
    smoothed_data = all_data[col].rolling(window=5, min_periods=1).mean()

    # 添加曲线 (使用分析系统的 Scatter 风格)
    fig.add_trace(
        go.Scatter(
            x=all_data.index, 
            y=smoothed_data, 
            name=col, 
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=1.8),
            connectgaps=True
        ), 
        row=i+1, col=1
    )
    
    # 优化坐标轴 (参考分析系统的网格和字体设置)
    fig.update_yaxes(
        title_text=col, 
        row=i+1, col=1, 
        title_font=dict(size=11),
        tickfont=dict(size=10),
        showgrid=True, 
        gridcolor='rgba(200,200,200,0.3)', # 轻微的网格线
        fixedrange=False,  # 允许缩放
        uirevision='constant', # 保证交互时数据状态的一致性
    )

# 统一配置 X 轴
fig.update_xaxes(
    showgrid=True, 
    gridcolor='rgba(200,200,200,0.3)',
    matches='x',
    tickfont=dict(size=10)
)

# ========== 4. 关键修改：布局配置 (完全同步分析系统风格) ==========
fig.update_layout(
    height=total_height, 
    title_text="<b>球磨机运行参数综合监测看板</b>",
    title_x=0.5,
    title_font=dict(size=20),
    showlegend=False,          # 既然每行都有标题，隐藏图例让画面更简洁
    template="plotly_white",   # 使用纯白底色，显得专业
    hovermode="x unified",     # 悬停联动
    margin=dict(l=80, r=40, t=100, b=80),
    dragmode='zoom',           # 开启框选放大
    plot_bgcolor='white'
)

# 在最后一个子图增加范围滑块 (参考分析系统)
fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.02), row=num_rows, col=1)

# ========== 5. 显示图表 ==========
fig.show()