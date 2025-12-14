#!/usr/bin/env python3
"""
Plot solver time analysis from planner_results.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKUP_CSV_FILE = os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/planner_results_backup.csv')
CSV_FILE = os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/planner_results.csv')

def parse_csv(csv_file):
    """Parse planner_results.csv file"""
    data = {}
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        n_device = int(row['n_device'])
        c_e = int(row['C_e'])
        time = float(row['time']) * 1000  # Convert to milliseconds
        
        if c_e not in data:
            data[c_e] = {'n_device': [], 'time': []}
        
        data[c_e]['n_device'].append(n_device)
        data[c_e]['time'].append(time)
    
    return data

def plot_time_analysis(data, output_path):
    """Plot time analysis chart"""
    
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams.update({
        'font.size': 7.5,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'lines.markersize': 4
    })
    
    fig, ax = plt.subplots(figsize=(3.17, 1.2))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (c_e, values) in enumerate(sorted(data.items())):
        # Sort by n_device for proper line connection
        sorted_indices = np.argsort(values['n_device'])
        n_device_sorted = [values['n_device'][idx] for idx in sorted_indices]
        time_sorted = [values['time'][idx] for idx in sorted_indices]
        
        ax.plot(n_device_sorted, time_sorted, 
                color=colors[i % len(colors)], 
                marker='o',
                label=f'C = {c_e}',
                markerfacecolor='white',
                markeredgewidth=1.0,
                linewidth=1.0,
                markersize=1)
    
    ax.set_xlabel('N', fontsize=7.5)
    ax.set_ylabel('Solving Time (ms)', fontsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([64, 128, 256, 512, 1024])
    ax.set_xticklabels(['64', '128', '256', '512', '1024'])
    ax.set_ylim(0.1, 1000)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(which='minor', length=0)
    ax.axhline(y=408.1875, color='grey', linestyle='--', linewidth=1.0, 
               alpha=0.8, label='Baseline Time')
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', 
              frameon=False, fancybox=False, edgecolor='none', shadow=False,
              ncol=2, fontsize=7.5)
    ax.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.6)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')

def main(type=None):
    """Main function"""
    if type == "default":
        csv_file = os.path.join(BASE_DIR, BACKUP_CSV_FILE)
    else:
        csv_file = os.path.join(BASE_DIR, CSV_FILE)
    data = parse_csv(csv_file)
    
    if not data:
        print("No data to plot")
        return
    
    print("Data summary:")
    for c_e in sorted(data.keys()):
        print(f"C_e = {c_e}: {len(data[c_e]['n_device'])} data points")
        print(f"  n_device range: {min(data[c_e]['n_device'])} - {max(data[c_e]['n_device'])}")
        print(f"  time range: {min(data[c_e]['time']):.3f} - {max(data[c_e]['time']):.3f} ms")
        print()
    
    plot_time_analysis(data, os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/figure_11.pdf'))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
