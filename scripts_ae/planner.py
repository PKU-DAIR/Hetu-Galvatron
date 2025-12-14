import numpy as np
import greedy_balancer as gb
import time
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_FILE = os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/planner_results.csv')

if len(sys.argv) > 1:
    device_list = [int(sys.argv[1])]
    C_e_list = [int(sys.argv[2])]
else:
    device_list = [64, 128, 256, 512, 1024]
    C_e_list = [1, 2, 4, 8, 16]
n_expert = 32

# Load existing CSV or create empty DataFrame
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded existing CSV with {len(df)} records")
else:
    df = pd.DataFrame(columns=['n_device', 'C_e', 'time'])
    print("Created new CSV file")

for C_e in C_e_list:
    for n_device in device_list:
        time_list = []
        for i in range(10):
            E = np.random.randint(0, 1000, (n_device, n_expert))
            start_time = time.time()
            _, _, A_res = gb.greedy_load_balancing_heuristic_complete(n_device, n_expert, E, C_e, 8192, 0)
            time_list.append(time.time() - start_time)
        avg_time = np.mean(time_list)
        print("n_device =", n_device, "C_e =", C_e, "time =", avg_time)
        
        # Check if entry exists
        mask = (df['n_device'] == n_device) & (df['C_e'] == C_e)
        
        if mask.any():
            # Update existing entry
            df.loc[mask, 'time'] = avg_time
            print(f"  Updated existing entry")
        else:
            # Append new entry
            new_row = pd.DataFrame({
                'n_device': [n_device],
                'C_e': [C_e],
                'time': [avg_time]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            print(f"  Added new entry")

# Save to CSV
df.to_csv(CSV_FILE, index=False)
print(f"\nResults saved to {CSV_FILE}")
