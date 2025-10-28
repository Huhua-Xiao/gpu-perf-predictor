#!/usr/bin/env python3
import subprocess
import pandas as pd
import re

def parse_ncu_output(output):
    metrics = {}

    
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
    all_data = parse_ncu_output(result.stdout)
    
    df = pd.DataFrame(all_data)
    df.to_csv("complete_data.csv", index=False)
    
    print("All data collected in single run!")

if __name__ == "__main__":
    main()
