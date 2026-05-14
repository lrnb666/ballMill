import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 加载数据
file_path = r'E:\ballMill\data\processed_1min_data.csv'
df = pd.read_csv(file_path)
df['time'] = pd.to_datetime(df['time'])

# 2. 列清单（严格对应你给出的参数）
cols = [
    'Current_A', 'Gear_IR_1', 'Gear_IR_2', 'Gear_IR_3', 
    'Particle_Size', 'Lub_Temp_Before', 'Lub_Temp_After', 
    'Bearing_In_1', 'Bearing_In_2', 'Bearing_In_3', 
    'Bearing_Out_2', 'Bearing_Out_3', 'Oil_Return_Temp', 
    'Vibrate_1', 'Vibrate_2', 'Vibrate_3', 'Vibrate_4'
]

# 3. 创建子图：行数 = 参数总数，每一行都是独立的 Y 轴
fig = make_subplots(
    rows=len(cols), 
    cols=1, 
    shared_xaxes=True,      # 保持时间轴同步，方便对比不同参数在同一时刻的状态
    vertical_spacing=0.015,  # 图与图之间的间距
    subplot_titles=cols     # 每个子图上方的参数名
)

# 4. 遍历参数，每个参数各画一个图
for i, col in enumerate(cols, start=1):
    if col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['time'], 
                y=df[col], 
                name=col, 
                mode='lines',
                line=dict(width=1)
            ),
            row=i, col=1
        )

# 5. 配置布局
fig.update_layout(
    height=300 * len(cols),    # 动态分配高度，每个参数给 300 像素，确保不拥挤
    title_text="球磨机运行参数实时趋势图 (全参数独立显示)",
    showlegend=False,          # 已经有子图标题了，不需要右侧图例
    template="plotly_white",
    margin=dict(t=80, b=50, l=50, r=50) # 调整边距
)

# 统一设置所有 Y 轴的属性（自动缩放以展示细节）
fig.update_yaxes(autorange=True, fixedrange=False)

# 6. 渲染到同一个页面
fig.show()