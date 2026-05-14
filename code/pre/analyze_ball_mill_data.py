import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================= 配置区 =================
CONFIG = {
    'main_dir': r'E:\ballMill',
    'input_file': r'output\processed_1min_data_0_to_400.csv',
    'output_csv': 'output\comprehensive_fault_analysis.csv',
    'thresholds': {
        'curr_running': 300,   # 满载运行电流基准
        'curr_stop': 50,      # 停机判定电流基准
        'vib_spike': 1.5,     # 振动平均变化阈值
        'temp_rise': 0.4,     # 轴承温升阈值
        'ir_imbalance': 8.0    # 红外温差标准差阈值
    }
}

# ================= 核心分析函数 =================
def analyze_and_plot_ball_mill():
    file_path = os.path.join(CONFIG['main_dir'], CONFIG['input_file'])
    print(f"正在分析文件: {file_path}...")
    
    if not os.path.exists(file_path):
        print("错误：未找到数据文件，请检查路径。")
        return

    # 1. 读取数据
    df = pd.read_csv(file_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

    # 2. 传感器分组
    temp_cols = ['Bearing_In_1', 'Bearing_In_2', 'Bearing_In_3', 'Bearing_Out_2', 'Bearing_Out_3']
    vib_cols = ['Vibrate_1', 'Vibrate_2', 'Vibrate_3', 'Vibrate_4']
    ir_cols = ['Gear_IR_1', 'Gear_IR_2', 'Gear_IR_3']

    # 3. 特征计算
    # 计算所有轴承中升温最快的一个点的变化率
    df['max_temp_rise'] = df[temp_cols].diff().max(axis=1)
    # 计算振动组的平均变化强度 (取绝对值均值更稳健)
    df['avg_vib_diff'] = df[vib_cols].diff().abs().mean(axis=1)
    # 计算红外温度分布的标准差
    df['ir_imbalance'] = df[ir_cols].std(axis=1)

    # 4. 判定逻辑
    is_running = df['Current_A'] > CONFIG['thresholds']['curr_running']
    
    # 逻辑A: 人为停机识别
    df['is_shutdown'] = (df['Current_A'] < 10) & (df['Current_A'].shift(1) > CONFIG['thresholds']['curr_stop'])
    
    # 逻辑B: 疑似故障分类
    df['is_fault_vibrate'] = is_running & (df['avg_vib_diff'] > CONFIG['thresholds']['vib_spike'])
    df['is_fault_temp'] = is_running & (df['max_temp_rise'] > CONFIG['thresholds']['temp_rise'])
    df['is_fault_ir'] = is_running & (df['ir_imbalance'] > CONFIG['thresholds']['ir_imbalance'])
    
    # 汇总异常
    df['any_fault'] = df['is_fault_vibrate'] | df['is_fault_temp'] | df['is_fault_ir']
    fault_points = df[df['any_fault']].copy()

    # 5. 输出统计
    print("-" * 30)
    print(f"分析完成！")
    print(f"正常停机事件数: {df['is_shutdown'].sum()}")
    print(f"疑似故障总点数: {len(fault_points)}")
    if len(fault_points) > 0:
        print(f" - 振动异常: {df['is_fault_vibrate'].sum()} 点")
        print(f" - 温升异常: {df['is_fault_temp'].sum()} 点")
        print(f" - 红外异常: {df['is_fault_ir'].sum()} 点")
        fault_points.to_csv(os.path.join(CONFIG['main_dir'], CONFIG['output_csv']))

    # 6. 交互式可视化
    print("生成可视化图表中...")
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("电流与异常标记", "平均振动变化强度 (Avg Vibration Diff)", "轴承最大温升率 (Max Temp Rise)")
    )

    # 子图1: 电流 + 故障红点
    fig.add_trace(go.Scatter(x=df.index, y=df['Current_A'], name='电流(A)', line=dict(color='#2E86C1')), row=1, col=1)
    if not fault_points.empty:
        fig.add_trace(go.Scatter(
            x=fault_points.index, y=fault_points['Current_A'], 
            mode='markers', name='异常点',
            marker=dict(color='red', size=7, symbol='x')
        ), row=1, col=1)

    # 子图2: 振动变化
    fig.add_trace(go.Scatter(x=df.index, y=df['avg_vib_diff'], name='振动变化强度', line=dict(color='#F39C12')), row=2, col=1)
    fig.add_hline(y=CONFIG['thresholds']['vib_spike'], line_dash="dash", line_color="red", row=2, col=1)

    # 子图3: 温度变化
    fig.add_trace(go.Scatter(x=df.index, y=df['max_temp_rise'], name='最大温升率', line=dict(color='#C0392B')), row=3, col=1)
    fig.add_hline(y=CONFIG['thresholds']['temp_rise'], line_dash="dash", line_color="red", row=3, col=1)

    fig.update_layout(height=850, title_text="球磨机多参数综合异常分析系统", showlegend=True)
    fig.show()

if __name__ == "__main__":
    analyze_and_plot_ball_mill()