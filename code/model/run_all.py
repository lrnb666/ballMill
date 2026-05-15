import os
import subprocess
import time

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 注意检查 TCN 文件夹下的文件名是不是 TCN.py
MODEL_SCRIPTS = [
    "GRU/GRU.py",
    "LSTM/LSTM.py",
    "TCN/TCN.py",               # 请根据实际文件名修改
    "LightGBM/LightGBM.py",
    "XGBoost/XGBoost.py",
    "CatBoost/CatBoost.py"
    "SMamba/SMamba.py",
    "PCMP-Mamba/PCMP_Mamba.py",
]

def run_all_models():
    print("="*60)
    print("🚀 开始一键自动化执行所有模型训练与评估")
    print("="*60)
    
    total_start_time = time.time()
    success_list = []
    failed_list = []

    for script in MODEL_SCRIPTS:
        if not os.path.exists(script):
            print(f"❌ 找不到文件: {script}，跳过该模型。")
            failed_list.append(script)
            continue

        model_name = script.split('/')[0]
        print(f"\n[{model_name}] 正在启动训练 >>> {script}")
        
        start_time = time.time()
        
        try:
            # 使用 subprocess 执行脚本
            # 这样每次运行都是一个独立的进程，跑完自动释放 GPU 显存！
            result = subprocess.run(
                ["python", script], 
                check=True,          # 如果脚本报错，会抛出 CalledProcessError
                text=True            # 将输出解码为字符串
            )
            
            elapsed = time.time() - start_time
            print(f"✅ [{model_name}] 执行完毕! 耗时: {elapsed:.2f} 秒")
            success_list.append(model_name)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ [{model_name}] 执行失败，发生错误！")
            failed_list.append(model_name)
        except KeyboardInterrupt:
            print("\n⚠️ 检测到手动中断(Ctrl+C)，停止执行后续模型。")
            break

    # 打印最终总结报告
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print(f"🎉 自动化测试流水线结束 | 总耗时: {total_time/60:.2f} 分钟")
    print(f"✅ 成功跑完的模型 ({len(success_list)}): {', '.join(success_list)}")
    if failed_list:
        print(f"❌ 运行失败的模型 ({len(failed_list)}): {', '.join(failed_list)}")
    print("="*60)
    print("提示: 最终的指标对比结果，请查看 Autils/eval_config 中配置的 txt 报告文件。")

if __name__ == "__main__":
    run_all_models()