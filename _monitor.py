"""训练监控脚本 - 实时显示训练进度、loss曲线、GPU状态"""
import os
import sys
import time
import glob
import csv
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_experiment():
    """找到最新的实验目录"""
    exp_dirs = sorted(glob.glob('experiments/exp_*'), reverse=True)
    return Path(exp_dirs[0]) if exp_dirs else None

def read_log_csv(csv_path):
    """读取训练日志CSV"""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def main():
    print("=" * 70)
    print("  训练实时监控器  Training Live Monitor")
    print("=" * 70)
    print("按 Ctrl+C 退出")
    print()

    prev_num_epochs = 0
    last_csv_mtime = 0

    while True:
        try:
            exp_dir = get_latest_experiment()
            if exp_dir is None:
                print("[等待] 还没有实验目录（训练可能还没启动）...")
                time.sleep(10)
                continue

            csv_path = exp_dir / 'training_log.csv'
            csv_mtime = os.path.getmtime(csv_path) if os.path.exists(csv_path) else 0

            if csv_mtime != last_csv_mtime:
                last_csv_mtime = csv_mtime
                rows = read_log_csv(csv_path)
                if not rows:
                    print(f"[等待] 实验目录: {exp_dir.name}，暂无训练日志...")
                    time.sleep(10)
                    continue

                clear_screen()
                print("=" * 70)
                print(f"  实验: {exp_dir.name}  |  GPU: RTX 4060 Laptop")
                print("=" * 70)
                print(f"{'Epoch':<8}{'Stage':<8}{'Train':<10}{'Val':<10}{'RPN_Cls':<10}{'RPN_BBox':<10}{'LR':<10}")
                print("-" * 70)

                for r in rows:
                    print(f"{r['Epoch']:<8}{r['Stage']:<8}{r['Train_Loss']:<10}{r['Val_Loss']:<10}"
                          f"{r['RPN_Cls_Loss']:<10}{r['RPN_BBox_Loss']:<10}{r['LR']:<10}")

                # 简单趋势分析
                print("=" * 70)
                if len(rows) >= 2:
                    last_train = float(rows[-1]['Train_Loss'])
                    prev_train = float(rows[-2]['Train_Loss'])
                    last_val = float(rows[-1]['Val_Loss'])
                    prev_val = float(rows[-2]['Val_Loss'])

                    train_trend = "↓" if last_train < prev_train else "↑"
                    val_trend = "↓" if last_val < prev_val else "↑"

                    print(f"趋势: Train={train_trend}  Val={val_trend}")
                    print(f"当前Epoch={rows[-1]['Epoch']}, 阶段={rows[-1]['Stage']}")

                print("=" * 70)
                print(f"按 Ctrl+C 退出 | 数据更新: {time.strftime('%H:%M:%S')}")
            else:
                # 文件未更新
                if rows:
                    sys.stdout.write(".")
                    sys.stdout.flush()

            time.sleep(15)  # 15秒检查一次

        except KeyboardInterrupt:
            print("\n\n监控已停止")
            break
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
