import pandas as pd
import warnings
import os

warnings.filterwarnings('ignore')

# ========== 核心修改：指定您想要处理的 Sheet 区间 ==========
# 注意：Python 从 0 开始数。 
# 0 到 10 代表第 1~10 个 Sheet；10 到 20 代表第 11~20 个 Sheet。（前闭后开）
START_DAY = 0 
END_DAY = 400  
RESAMPLE_FREQ = '1min' 

# ========== 配置路径与参数 ==========
main_dir = r'E:\ballMill'
data_dir = os.path.join(main_dir, 'data')
output_dir = os.path.join(main_dir, 'output')
# 【动态命名】让生成的文件带上天数区间，防止覆盖之前的 0-10 天数据
file_suffix = f"{START_DAY}_to_{END_DAY}"
output_file = os.path.join(output_dir, f'processed_1min_data_{file_suffix}.csv')

cache_dir = os.path.join(output_dir, 'pre','cache','cache_1min_data')
os.makedirs(cache_dir, exist_ok=True) 

FORCE_REPROCESS = False 

file_map = {
    'GBS_MC_QMJ_101_DL_REAL(1#球磨机101电流).xlsx': 'Current_A',
    'GBS_MC_Fear_Infrared_1_MW(1#球磨机大小齿轮红外测温 1).xlsx': 'Gear_IR_1',
    'GBS_MC_Fear_Infrared_2_MW(1#球磨机大小齿轮红外测温 2).xlsx': 'Gear_IR_2',
    'GBS_MC_Fear_Infrared_3_MW(1#球磨机大小齿轮红外测温 3).xlsx': 'Gear_IR_3',
    'GBS_MC_LD1_REAL(1#球磨粒度).xlsx': 'Particle_Size',
    'GBS_MC_LUB_T_Before_Coolwater(低压循环冷却前温度).xlsx': 'Lub_Temp_Before', 
    'GBS_MC_LUB_T_Behind_Coolwater(低压循环冷却后温度).xlsx': 'Lub_Temp_After',
    'GBS_MC_N1BearingTemp_IW(进料端主轴测温1).xlsx': 'Bearing_In_1',
    'GBS_MC_N2BearingTemp_IW(进料端主轴测温2).xlsx': 'Bearing_In_2',
    'GBS_MC_N3BearingTemp_IW(进料端主轴测温3).xlsx': 'Bearing_In_3',
    'GBS_MC_N5BearingTemp_IW(出料端主轴测温2).xlsx': 'Bearing_Out_2',
    'GBS_MC_N6BearingTemp_IW(出料端主轴测温3).xlsx': 'Bearing_Out_3',
    'GBS_MC_OilImportTemp_IW(回油口温转换).xlsx': 'Oil_Return_Temp',
    'GBS_MC_Pinion_Vibrate_1_MW(小齿轮震动检测1).xlsx': 'Vibrate_1',
    'GBS_MC_Pinion_Vibrate_2_MW(小齿轮震动检测2).xlsx': 'Vibrate_2',
    'GBS_MC_Pinion_Vibrate_3_MW(小齿轮震动检测3).xlsx': 'Vibrate_3',
    'GBS_MC_Pinion_Vibrate_4_MW(小齿轮震动检测4).xlsx': 'Vibrate_4',
}

def get_sensor_data(file_name, col_name):
    print(f"  -> 🔍 分析任务: {col_name}")
    path = os.path.join(data_dir, file_name)
    xl = pd.ExcelFile(path)
    target_sheet_names = xl.sheet_names[START_DAY : END_DAY]
    
    if not target_sheet_names:
        raise ValueError(f"区间 [{START_DAY}:{END_DAY}] 无数据")
        
    df_list = []
    for sheet_name in target_sheet_names:
        day_cache_file = os.path.join(cache_dir, f"{col_name}_{sheet_name}.csv")
        
        if os.path.exists(day_cache_file) and not FORCE_REPROCESS:
            daily_resampled_df = pd.read_csv(day_cache_file, index_col=0, parse_dates=True)
        else:
            # 只读取需要的列，进一步提速
            df = pd.read_excel(xl, sheet_name=sheet_name, usecols=['创建时间', 'value'])
            df['time'] = pd.to_datetime(df['创建时间']).dt.tz_localize(None)
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            daily_resampled_df = df.set_index('time')[['value']].resample(RESAMPLE_FREQ).mean()
            daily_resampled_df.to_csv(day_cache_file)
            
        df_list.append(daily_resampled_df)
        
    combined_df = pd.concat(df_list)
    combined_df.columns = [col_name]
    # 去除重复索引（万一 Sheet 之间有重叠）
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    return combined_df

    
# ========== 执行 ==========
print(f"========== 开始数据预处理任务 ==========")
print(f"当前处理区间: 第 {START_DAY} 个 到 第 {END_DAY} 个 Sheet")

all_data = None
total_files = len(file_map)

for i, (file, col_name) in enumerate(file_map.items(), 1):
    print(f"[{i}/{total_files}] 传感器: {col_name}")
    try:
        df = get_sensor_data(file, col_name)
        if all_data is None:
            all_data = df
        else:
            all_data = all_data.join(df, how='outer')
    except Exception as e:
        print(f"  -> ❌ 错误: {e}")

print("\n正在对齐并保存最终数据...")
all_data.sort_index(inplace=True)
all_data.ffill(inplace=True) 
all_data.to_csv(output_file)

print(f" 区间 [{START_DAY} 到 {END_DAY}] 预处理完成！")
print(f"文件保存在: {output_file}")