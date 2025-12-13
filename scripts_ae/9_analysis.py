#!/usr/bin/env python3
"""
Training Log Analysis Script - Data Analysis Part
Parse and analyze training log files, extract statistics
"""

import re
import os
import glob
import pandas as pd
import numpy as np

def parse_megatron_log(filepath):
    """Parse Megatron format training log file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                pattern = r'iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s+([\d.]+).*?lm loss:\s+([\d.eE+-]+)'
                match = re.search(pattern, line)
                if match:
                    iteration = int(match.group(1))
                    time_ms = float(match.group(3))
                    loss_str = match.group(4)
                    
                    try:
                        loss = float(loss_str)
                    except ValueError:
                        continue
                    
                    data.append({
                        'iteration': iteration,
                        'loss': loss,
                        'time_ms': time_ms
                    })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    return data

def parse_galvatron_log(filepath):
    """Parse Galvatron format training log file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                pattern = r'\| Iteration:\s+(\d+)\s+\|.*?Elapsed time per iteration \(ms\):\s+([\d.]+)\s+\|.*?Loss:\s+([\d.e+-]+)'
                match = re.search(pattern, line)
                if match:
                    iteration = int(match.group(1))
                    time_ms = float(match.group(2))
                    loss = float(match.group(3))
                    
                    data.append({
                        'iteration': iteration,
                        'loss': loss,
                        'time_ms': time_ms
                    })
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    return data

def parse_log_file(filepath):
    """Automatically detect and parse log file format."""
    data = parse_galvatron_log(filepath)
    if data:
        return data
    return parse_megatron_log(filepath)

def parse_filename(filename):
    """Parse log filename to extract configuration (similar to 8_analysis.py)."""
    # ${METHOD}_${MODEL_NAME}_${DATASET}_batch${BATCH_SIZE}_seq${SEQ_LENGTH}_aux${AUX_LOSS}_${ARNOLD_ID}.log
    pattern = r'(.+)_convergence_(.+)_(.+)_batch(\d+)_seq(\d+)_aux(.+)_(\d+)\.log'
    match = re.match(pattern, filename)
    
    if match:
        if int(match.group(5)) != 4096:
            return None
        return {
            'method': match.group(1),
            'model_name': match.group(2),
            'dataset': match.group(3),
            'aux_loss': float(match.group(6)),
            'filename': filename
        }
    return None
def process_log_files(filepaths):
    """Process multiple log files and return analyzed data."""
    all_data = []
    
    for filepath in filepaths:
        print(f"Processing: {filepath}")
        
        # Parse filename to extract configuration
        filename = os.path.basename(filepath)
        config = parse_filename(filename)
        if not config:
            continue
        
        method = config['method']
        aux_loss = config['aux_loss']
        
        # Parse log file content
        data = parse_log_file(filepath)
        
        if not data:
            continue
        
        # Add method and aux_loss to each data point
        for item in data:
            all_data.append({
                'iter': item['iteration'],
                'method': method,
                'aux_loss': aux_loss,
                'loss': item['loss'],
                'time': item['time_ms']
            })
        
        print(f"  Found {len(data)} iterations (method={method}, aux_loss={aux_loss})")
    
    return all_data

def find_log_files(directories):
    """Find all log files in given directories (similar to 8_analysis.py)."""
    all_files = []
    for directory in directories:
        if os.path.isdir(directory):
            log_files = glob.glob(os.path.join(directory, "*.log"))
            all_files.extend(log_files)
            print(f"Found {len(log_files)} log files in {directory}")
        elif os.path.isfile(directory):
            # If it's a file, add it directly
            all_files.append(directory)
    return all_files

def main():
    """Main analysis function."""
    import sys
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LAER_LOG_DIR = os.path.join(BASE_DIR, 'LAER-MoE', 'galvatron', 'models', 'moe', 'training_log')
    MEGATRON_LOG_DIR = os.path.join(BASE_DIR, 'Megatron', 'training_log')

    directories = []
    if os.path.exists(LAER_LOG_DIR):
        directories.append(LAER_LOG_DIR)
    if os.path.exists(MEGATRON_LOG_DIR):
        directories.append(MEGATRON_LOG_DIR)
    
    filepaths = find_log_files(directories)

    all_data = process_log_files(filepaths)
    
    # Print summary
    if all_data:
        df = pd.DataFrame(all_data)
        print("\n" + "="*60)
        print("Analysis Summary")
        print("="*60)
        print(f"Total records: {len(df)}")
        print(f"Methods: {df['method'].unique()}")
        print(f"Aux losses: {sorted(df['aux_loss'].unique())}")
        print(f"Iteration range: {df['iter'].min()} - {df['iter'].max()}")
    
    # Save results to CSV for plotting
    if all_data:
        output_file = BASE_DIR + '/LAER-MoE/scripts_ae/convergence_analysis.csv'
        df = pd.DataFrame(all_data)
        # Ensure column order: iter, method, aux_loss, loss, time
        df = df[['iter', 'method', 'aux_loss', 'loss', 'time']]
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

