import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CONFIG = {
    'main_dir': r'E:\ballMill',
    'input_file': r'output\processed_1min_data_0_to_400.csv',
    'thresholds': {
        'curr_drop': -40,    # 电流一分钟突降阈值
        'vib_spike': 1.2,    # 震动突增阈值
        'temp_rise': 0.3     # 温升速率阈值
    }
}

def analyze_and_annotate_all_cases():
    file_path = os.path.join(CONFIG['main_dir'], CONFIG['input_file'])
    df = pd.read_csv(file_path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

    temp_cols = ['Bearing_In_1', 'Bearing_In_2', 'Bearing_In_3', 'Bearing_Out_2', 'Bearing_Out_3']
    vib_cols = ['Vibrate_1', 'Vibrate_2', 'Vibrate_3', 'Vibrate_4']

    # 1. 基础特征计算
    df['curr_diff'] = df['Current_A'].diff()
    df['avg_vib_diff'] = df[vib_cols].diff().abs().mean(axis=1)
    df['max_temp_rise'] = df[temp_cols].diff().max(axis=1)

    # 2. 回溯机制：用滚动窗口寻找前5分钟是否有异常发生
    # 使用 rolling(5).max()，意味着只要过去5分钟内出现过一次尖峰，就算数
    df['recent_vib_anomaly'] = (df['avg_vib_diff'] > CONFIG['thresholds']['vib_spike']).rolling(5).max() > 0
    df['recent_temp_anomaly'] = (df['max_temp_rise'] > CONFIG['thresholds']['temp_rise']).rolling(5).max() > 0

    # 3. 场景判定逻辑
    was_running = df['Current_A'].rolling(5).mean() > 200
    is_curr_crash = was_running & (df['curr_diff'] <= CONFIG['thresholds']['curr_drop'])
    is_normal_stop = (df['Current_A'] < 10) & (df['Current_A'].shift(1) > 50)

    df['case_type'] = "正常运行"
    
    # 场景 A: 正常停机 (电流掉落，且没有先导报警)
    mask_normal_stop = is_normal_stop & ~df['recent_vib_anomaly'] & ~df['recent_temp_anomaly']
    df.loc[mask_normal_stop, 'case_type'] = "正常人为关机"

    # 场景 B/C: 机械故障导致的跳闸 (电流骤降，且之前5分钟内震动或温度报警过)
    mask_mech_crash = is_curr_crash & (df['recent_vib_anomaly'] | df['recent_temp_anomaly'])
    df.loc[mask_mech_crash, 'case_type'] = "机械故障跳闸 (伴随震动/高温)"

    # 场景 D: 纯电气异常 (电流骤降，但机械指标完全正常)
    mask_elec_crash = is_curr_crash & ~df['recent_vib_anomaly'] & ~df['recent_temp_anomaly']
    df.loc[mask_elec_crash, 'case_type'] = "纯电气骤降 (无机械异常)"

    # 4. 可视化绘制
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        subplot_titles=("电流趋势及事件标注", "震动强度监控", "轴承温升监控"))

    # 画底线
    fig.add_trace(go.Scatter(x=df.index, y=df['Current_A'], name='电流(A)', line=dict(color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['avg_vib_diff'], name='震动强度', line=dict(color='#85C1E9')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['max_temp_rise'], name='最大温升', line=dict(color='#F5B041')), row=3, col=1)

    # 提取事件点并打上特殊标记
    events = df[df['case_type'] != "正常运行"]
    
    for event_type, color, symbol in [
        ("正常人为关机", "green", "circle"),
        ("机械故障跳闸 (伴随震动/高温)", "red", "x"),
        ("纯电气骤降 (无机械异常)", "purple", "triangle-down")
    ]:
        subset = events[events['case_type'] == event_type]
        if not subset.empty:
            # 在图表上画点
            fig.add_trace(go.Scatter(
                x=subset.index, y=subset['Current_A'], mode='markers',
                marker=dict(color=color, size=10, symbol=symbol), name=event_type
            ), row=1, col=1)
            
            # 在图表上直接写文字标注 (选前几个点展示，防止文字重叠)
            for idx in subset.index[:10]:
                fig.add_annotation(
                    x=idx, y=subset.loc[idx, 'Current_A'],
                    text=event_type.split(" ")[0], showarrow=True, arrowhead=1,
                    ax=0, ay=-40, font=dict(color=color, size=12), row=1, col=1
                )

    # 画出阈值红线
    fig.add_hline(y=CONFIG['thresholds']['vib_spike'], line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=CONFIG['thresholds']['temp_rise'], line_dash="dash", line_color="red", row=3, col=1)

    fig.update_layout(height=900, title="球磨机全场景故障诊断图", hovermode="x unified")
    fig.show()

if __name__ == "__main__":
    analyze_and_annotate_all_cases()