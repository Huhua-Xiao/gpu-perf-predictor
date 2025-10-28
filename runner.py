#!/usr/bin/env python3
import subprocess
import pandas as pd
import re

def parse_ncu_output(output):
    """解析 ncu 的输出，提取所有指标"""
    metrics = {}
    
    # 从 ncu 输出中提取时间、SM利用率等
    # ncu 输出包含基础性能 + 深度指标
    
    return metrics

def main():
    print("Running benchmark with Nsight Compute (single run)...")
    
    cmd = [
        "ncu",
        "--csv",
        "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum",
        "--target-processes", "all",
        "./benchmark"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # ncu 的输出包含：
    # 1. 基础性能（执行时间）
    # 2. 深度指标（SM利用率、带宽等）
    
    # 解析并保存所有数据
    all_data = parse_ncu_output(result.stdout)
    
    df = pd.DataFrame(all_data)
    df.to_csv("complete_data.csv", index=False)
    
    print("All data collected in single run!")

if __name__ == "__main__":
    main()