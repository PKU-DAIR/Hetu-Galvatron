import re
import os
import glob
from collections import defaultdict
import numpy as np

START_ITER = 21
END_ITER = 70
LAER_LOG_DIR = 'LAER-MoE/galvatron/models/moe/training_log'
MEGATRON_LOG_DIR = 'Megatron/training_log'

def parse_filename(filename):
    """Parse log filename to extract configuration."""
    # ${METHOD}_${MODEL_NAME}_${DATASET}_batch${BATCH_SIZE}_seq${SEQ_LENGTH}_aux${AUX_LOSS}_${ARNOLD_ID}.log
    pattern = r'(.+)_(.+)_(.+)_batch(\d+)_seq(\d+)_aux(.+)_(\d+)\.log'
    match = re.match(pattern, filename)
    
    if match:
        if int(match.group(5)) != 8192:
            return None
        return {
            'method': match.group(1),
            'model_name': match.group(2),
            'dataset': match.group(3),
            'aux_loss': float(match.group(6)),
            'filename': filename
        }
    return None

def parse_log_line(line, is_megatron=False):
    """Extract iteration and elapsed time from log line."""
    if is_megatron:
        pattern = r'\[.*?\] iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+)'
    else:
        pattern = r'\| Iteration:\s+(\d+)\s+\|\s+Consumed samples:\s+(\d+)\s+\|\s+Elapsed time per iteration \(ms\):\s+([\d.]+)\s+\|\s+Learning rate:\s+([\d.]+e-\d+)\s+\|\s+Loss:\s+([\d.]+e\+\d+)\s+\|\s+grad norm:\s+([\d.]+)'
    match = re.search(pattern, line)
    
    if match:
        return int(match.group(1)), int(match.group(2)), float(match.group(3))
    return None, None, None

def analyze_log_file(filepath, is_megatron=False):
    """Analyze a single log file and extract timing statistics."""
    config = parse_filename(os.path.basename(filepath))
    if not config:
        return None
    
    iterations = []
    elapsed_times = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                iteration, _, elapsed_time = parse_log_line(line, is_megatron)
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

def analyze_directory(log_dir, is_megatron=False):
    """Analyze all log files in a directory."""
    log_files = glob.glob(f"{log_dir}/*.log")
    
    if not log_files:
        print(f"No .log files found in {log_dir}")
        return []
    
    print(f"Found {len(log_files)} log files in {log_dir}")
    
    results = []
    for log_file in log_files:
        result = analyze_log_file(log_file, is_megatron)
        if result:
            results.append(result)
    
    return results

def save_to_csv(results, filename):
    """Save results to CSV file."""
    try:
        import pandas as pd
        
        csv_data = []
        for result in results:
            csv_data.append({
                'dataset': result['dataset'],
                'method': result['method'],
                'model_name': result['model_name'],
                'aux_loss': result['aux_loss'],
                'avg_time_ms': result['avg_time'],
                'std_time_ms': result.get('std_time', 0),
                'matched_iterations': result['matched_iterations'],
                'total_iterations': result['total_iterations']
            })
        
        csv_data.sort(key=lambda x: (x['method'], x['model_name'], x['aux_loss'], x['dataset']))
        
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

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    laer_log_dir = os.path.join(BASE_DIR, LAER_LOG_DIR)
    megatron_log_dir = os.path.join(BASE_DIR, MEGATRON_LOG_DIR)
    # Analyze both directories
    all_results = []
    
    if os.path.exists(laer_log_dir):
        print(f"\nAnalyzing {laer_log_dir}...")
        results = analyze_directory(laer_log_dir, is_megatron=False)
        all_results.extend(results)
    
    if os.path.exists(megatron_log_dir):
        print(f"\nAnalyzing {megatron_log_dir}...")
        results = analyze_directory(megatron_log_dir, is_megatron=True)
        all_results.extend(results)
    
    if not all_results:
        print("\nNo valid results found")
        return
    
    print(f"\nAnalysis complete: {len(all_results)} valid files processed")

    # Save to CSV
    CSV_DIR = os.path.join(BASE_DIR, 'LAER-MoE', 'scripts_ae')
    if all_results:
        csv_filename = CSV_DIR + f'/e2e_analysis.csv'
        save_to_csv(all_results, csv_filename)
        print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()
