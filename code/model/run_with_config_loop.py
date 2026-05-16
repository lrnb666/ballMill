import os
import re
import subprocess
import time

# =====================【多组参数配置：在这里加组！】=====================
# 格式：(PREDICT_HORIZON, INDUSTRIAL_HIT_REL_TOLERANCE, LAG_STEPS)
PARAM_SETS = [
   # (5,   0.005, 60),   # 第1组
     (5,   0.001, 60),   # 第1组
     (10,   0.001, 60), 
     (15,   0.001, 60), 
     (20,   0.001, 60), 

    # (5,   0.005, 60), 
     (10,   0.005, 60), 
    (15,   0.005, 60), 
     (20,   0.005, 60), 
    # (10,  0.005, 60),   # 第2组（需要就解开注释）
    # (5,   0.01,  60),   # 第3组
    # (15,  0.003, 60),   # 第4组
]
# ======================================================================

# 固定路径
CONFIG_FILE = "/home/lr/projects/ballMill/code/model/Autils/eval_config.py"
RUN_ALL_SCRIPT = "/home/lr/projects/ballMill/code/model/run_all.py"

def update_config_file(pred_horizon, tolerance, lag_steps):
    """自动更新 eval_config.py 参数"""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. 更新预测步长
    content = re.sub(
        r"PREDICT_HORIZON\s*=\s*\d+",
        f"PREDICT_HORIZON = {pred_horizon}",
        content
    )
    content = re.sub(
        r"PRED_STEPS\s*=\s*PREDICT_HORIZON",
        f"PRED_STEPS = PREDICT_HORIZON",
        content
    )

    # 2. 更新工业容忍度
    content = re.sub(
        r"INDUSTRIAL_HIT_REL_TOLERANCE\s*=\s*[\d\.]+",
        f"INDUSTRIAL_HIT_REL_TOLERANCE = {tolerance}",
        content
    )

    # 3. 自动生成报告文件名
    report_name = f"model_eval_compare_{lag_steps}_{pred_horizon}_{tolerance}.txt"
    content = re.sub(
        r'METRICS_REPORT_PATH\s*=\s*str\(_PROJECT_ROOT\s*/\s*"output"\s*/\s*".*?"\)',
        f'METRICS_REPORT_PATH = str(_PROJECT_ROOT / "output" / "{report_name}")',
        content
    )

    # 保存
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n✅ 已切换参数组：")
    print(f"   LAG_STEPS          = {lag_steps}")
    print(f"   PREDICT_HORIZON    = {pred_horizon}")
    print(f"   误差容忍度         = {tolerance}")
    print(f"   输出报告           = {report_name}")

def run_all_models():
    """运行你的批量脚本"""
    os.chdir(os.path.dirname(RUN_ALL_SCRIPT))
    print("\n🚀 开始执行全部模型训练...\n")
    try:
        subprocess.run(["python", RUN_ALL_SCRIPT], check=True)
    except subprocess.CalledProcessError:
        print("❌ 本组模型执行失败，继续下一组...")
    except KeyboardInterrupt:
        print("\n⚠️ 手动中断，退出循环")
        exit(1)

if __name__ == "__main__":
    print("=" * 70)
    print("🔥 多参数组自动循环执行流水线")
    print(f"📌 共 {len(PARAM_SETS)} 组参数等待执行")
    print("=" * 70)

    total_start = time.time()

    for idx, (ph, tol, lag) in enumerate(PARAM_SETS, 1):
        print(f"\n\n==================================================")
        print(f"  正在执行第 {idx}/{len(PARAM_SETS)} 组参数")
        print(f"==================================================")

        # 1. 更新配置
        update_config_file(ph, tol, lag)

        # 2. 运行所有模型
        run_all_models()

        # 3. 间隔一下，释放显存
        time.sleep(2)

    print("\n" + "=" * 70)
    print(f"🎉 全部参数组执行完毕！总耗时：{(time.time() - total_start)/60:.1f} 分钟")
    print("=" * 70)
