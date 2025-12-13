#!/usr/bin/env python3
"""
Parse profile files and extract breakdown times
Output CSV: rank, model_name, method, alltoall_time, expert_time, total_time
"""

import os
import re
import glob
import sys
import pandas as pd

def parse_time_string(time_str):
    """Parse time string and convert to seconds"""
    if time_str.endswith('s'):
        return float(time_str[:-1])
    elif time_str.endswith('ms'):
        return float(time_str[:-2]) / 1000.0
    elif time_str.endswith('us'):
        return float(time_str[:-2]) / 1000000.0
    elif time_str.endswith('ns'):
        return float(time_str[:-2]) / 1000000000.0
    else:
        try:
            return float(time_str)
        except ValueError:
            return 0.0

def parse_profile_file(filepath):
    """Parse single profile file and extract key metrics"""
    a2a_cuda_total = 0.0
    mlp_cuda_total = 0.0
    profiler_step_cpu_total = 0.0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('---') or line.startswith('Name'):
                    continue
                
                parts = line.split()
                if len(parts) < 10:
                    continue
                
                operation_name = parts[0]
                if operation_name == "_AllToAll_with_event":
                    try:
                        a2a_cuda_total = parse_time_string(parts[8])
                    except (ValueError, IndexError):
                        continue
                elif operation_name == "expert_computation_time":
                    try:
                        mlp_cuda_total = parse_time_string(parts[8])
                    except (ValueError, IndexError):
                        continue
                elif "ProfilerStep" in operation_name:
                    try:
                        profiler_step_cpu_total = parse_time_string(parts[4])
                    except (ValueError, IndexError):
                        continue
        
        return {
            'a2a_cuda_total': a2a_cuda_total,
            'mlp_cuda_total': mlp_cuda_total,
            'profiler_step_cpu_total': profiler_step_cpu_total
        }
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return {'a2a_cuda_total': 0.0, 'mlp_cuda_total': 0.0, 'profiler_step_cpu_total': 0.0}

def parse_filename(filename):
    """Parse log filename to extract configuration (similar to 8_analysis.py)."""
    # ${METHOD}_${MODEL_NAME}_${DATASET}_batch${BATCH_SIZE}_seq${SEQ_LENGTH}_aux${AUX_LOSS}_${ARNOLD_ID}.log
    pattern = r'(.+)_breakdown_(.+)_(.+)_batch(\d+)_seq(\d+)_aux(.+)_(\d+)\.txt'
    match = re.match(pattern, filename)
    
    if match:
        return int(match.group(7)), match.group(2), match.group(1)
    return -1, None, None

def find_profile_files(directories):
    """Find all profile files in given directories"""
    all_files = []
    for directory in directories:
        if os.path.isdir(directory):
            profile_files = glob.glob(os.path.join(directory, "*.txt"))
            all_files.extend(profile_files)
            print(f"Found {len(profile_files)} profile files in {directory}")
        elif os.path.isfile(directory):
            all_files.append(directory)
    return all_files

def process_profile_files(filepaths):
    """Process profile files and return records"""
    records = []
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        rank, model_name, method = parse_filename(filename)
        
        if rank == -1:
            print(f"Warning: Could not extract rank from {filename}, skipping")
            continue
        
        file_data = parse_profile_file(filepath)
        alltoall_time = file_data['a2a_cuda_total']
        expert_time = file_data['mlp_cuda_total'] * 2
        total_time = file_data['profiler_step_cpu_total']
        
        if total_time > 0:
            records.append({
                'rank': rank,
                'model_name': model_name,
                'method': method,
                'alltoall_time': alltoall_time,
                'expert_time': expert_time,
                'total_time': total_time
            })
    
    return records

def main():
    """Main function"""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    directories = [
        os.path.join(BASE_DIR, 'LAER-MoE', 'galvatron', 'models', 'moe', 'training_log', 'breakdown'),
    ]
    
    output_csv = os.path.join(BASE_DIR, 'LAER-MoE', 'scripts_ae', 'breakdown_analysis.csv')
    
    print("Starting to parse profile files...")
    filepaths = find_profile_files(directories)
    
    if not filepaths:
        print("No profile files found")
        return
    
    # Process files
    records = process_profile_files(filepaths)
    
    if not records:
        print("No valid profile data found")
        return
    
    # Save to CSV
    df = pd.DataFrame(records)
    df = df.sort_values(['model_name', 'method', 'rank'])
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved {len(records)} records to {output_csv}")
    print(f"Models: {df['model_name'].unique().tolist()}")
    print(f"Methods: {df['method'].unique().tolist()}")
    print(f"Ranks: {sorted(df['rank'].unique().tolist())}")

if __name__ == "__main__":
    main()
