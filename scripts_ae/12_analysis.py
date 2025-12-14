#!/usr/bin/env python3
"""
Parse training log files and extract iteration times
Output CSV: method, avg_time_ms
"""

import os
import re
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(BASE_DIR, 'LAER-MoE', 'galvatron', 'models', 'moe', 'training_log', 'ablation')
START_ITER = 21
END_ITER = 70

def parse_filename(filename):
    """Parse log filename to extract method."""
    # Format: ${METHOD}_${MODEL_NAME}_${DATASET}_batch${BATCH_SIZE}_seq${SEQ_LENGTH}_aux${AUX_LOSS}_${ARNOLD_ID}.log
    pattern = r'(.+)_(.+)_(.+)_batch(\d+)_seq(\d+)_aux(.+)_(\d+)\.log'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'method': match.group(1),
            'filename': filename
        }
    return None

def parse_log_line(line):
    """Extract iteration and elapsed time from log line."""
    pattern = r'\| Iteration:\s+(\d+)\s+\|\s+Consumed samples:\s+(\d+)\s+\|\s+Elapsed time per iteration \(ms\):\s+([\d.]+)'
    match = re.search(pattern, line)
    
    if match:
        return int(match.group(1)), float(match.group(3))
    return None, None

def analyze_log_file(filepath):
    """Analyze a single log file and extract timing statistics."""
    config = parse_filename(os.path.basename(filepath))
    if not config:
        return None
    
    iterations = []
    elapsed_times = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                iteration, elapsed_time = parse_log_line(line)
                if iteration is not None:
                    iterations.append(iteration)
                    elapsed_times.append(elapsed_time)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    
    # Calculate average time for target iteration range
    target_iterations = list(range(START_ITER, END_ITER + 1))
    target_times = [elapsed_times[iterations.index(iter)] 
                    for iter in target_iterations if iter in iterations]
    
    if target_times:
        config.update({
            'avg_time': np.mean(target_times),
            'std_time': np.std(target_times),
            'matched_iterations': len(target_times),
            'total_iterations': len(target_iterations)
        })
        return config
    return None

def analyze_directory(log_dir):
    """Analyze all log files in a directory."""
    log_files = glob.glob(f"{log_dir}/*.log")
    
    if not log_files:
        print(f"No .log files found in {log_dir}")
        return []
    
    print(f"Found {len(log_files)} log files in {log_dir}")
    
    results = []
    for log_file in log_files:
        result = analyze_log_file(log_file)
        if result:
            results.append(result)
            print(f"Processed {os.path.basename(log_file)}: avg_time={result['avg_time']:.1f}ms")
    
    return results

def save_to_csv(results, filename):
    """Save results to CSV file."""
    try:
        csv_data = []
        for result in results:
            csv_data.append({
                'method': result['method'],
                'avg_time_ms': round(result['avg_time'], 2)
            })
        
        csv_data.sort(key=lambda x: x['method'])
        
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        
    except ImportError:
        print("\nWarning: pandas not installed, cannot save CSV")
        print("Install with: pip install pandas")

def main():
    """Main function."""
    print("Training Log Analysis Tool")
    print("="*50)
    print(f"Iteration range: {START_ITER}-{END_ITER}")
    print("="*50)
    
    if not os.path.exists(LOG_DIR):
        print(f"\nDirectory not found: {LOG_DIR}")
        return
    
    print(f"\nAnalyzing {LOG_DIR}...")
    all_results = analyze_directory(LOG_DIR)
    
    if not all_results:
        print("\nNo valid results found")
        return
    
    print(f"\nAnalysis complete: {len(all_results)} valid files processed")
    
    # Save to CSV
    CSV_DIR = os.path.join(BASE_DIR, 'LAER-MoE', 'scripts_ae')
    if all_results:
        csv_filename = os.path.join(CSV_DIR, 'ablation_analysis.csv')
        save_to_csv(all_results, csv_filename)
        
        df = pd.DataFrame(all_results)
        print(f"\nMethods: {df['method'].unique().tolist()}")

if __name__ == "__main__":
    main()


