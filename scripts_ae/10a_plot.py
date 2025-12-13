#!/usr/bin/env python3
"""
Plot breakdown chart from CSV file
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_FILE = 'LAER-MoE/scripts_ae/breakdown_analysis.csv'
BACKUP_CSV_FILE = 'LAER-MoE/scripts_ae/breakdown_analysis_backup.csv'

def load_data(csv_file):
    """Load data from CSV file"""
    if not os.path.exists(csv_file):
        backup_file = os.path.join(BASE_DIR, BACKUP_CSV_FILE)
        if os.path.exists(backup_file):
            csv_file = backup_file
            print(f"Using backup CSV: {backup_file}")
        else:
            print(f"Error: CSV file {csv_file} not found")
            return None
    
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['rank', 'model_name', 'method', 'alltoall_time', 'expert_time', 'total_time']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        print(f"Loaded {len(df)} records from {csv_file}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def prepare_plot_data(df):
    """Prepare data for plotting by calculating averages"""
    plot_data_16 = []
    plot_data_8 = []
    
    for (model_name, method), group in df.groupby(['model_name', 'method']):
        avg_alltoall = group['alltoall_time'].mean()
        avg_expert = group['expert_time'].mean()
        avg_total = group['total_time'].mean()
        other_time = avg_total - avg_alltoall - avg_expert
        
        plot_item = {
            'a2a': avg_alltoall,
            'mlp': avg_expert,
            'other': other_time,
            'profiler_time': avg_total,
            'method': method
        }
        
        if model_name == 'Mixtral-8x7b-e16k4':
            plot_data_16.append(plot_item)
        elif model_name == 'Mixtral-8x7b-e8k2':
            plot_data_8.append(plot_item)
    
    solver_order = ['NONE', 'FLEX', 'FSEP']
    def sort_by_method(data_list):
        return sorted(data_list, key=lambda x: solver_order.index(x['method']) if x['method'] in solver_order else 999)
    
    plot_data_16 = sort_by_method(plot_data_16)
    plot_data_8 = sort_by_method(plot_data_8)
    
    label_mapping = {'NONE': 'FSDP+EP', 'FLEX': 'FlexMoE', 'FSEP': 'LAER-MoE'}
    labels_16 = [label_mapping[d['method']] for d in plot_data_16]
    labels_8 = [label_mapping[d['method']] for d in plot_data_8]
    
    return plot_data_16, plot_data_8, labels_16, labels_8

def add_percentage_labels(ax, bars, times, bottom_positions, total_times, labels=None, plot_data=None, white=False):
    """Add percentage labels and speedup ratio on bars"""
    for i, (bar, time, bottom, total) in enumerate(zip(bars, times, bottom_positions, total_times)):
        height = bar.get_height()
        if height > 0:
            percentage = (time / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bottom + height/2,
                   f'{percentage:.1f}%', ha='center', va='center', fontsize=5, fontweight='bold',
                   color='black' if not white else 'white')
            
            if labels and plot_data and white:
                current_label = labels[i]
                if current_label in ['FSDP+EP', 'FlexMoE']:
                    laer_moe_data = None
                    for j, data in enumerate(plot_data):
                        if labels[j] == 'LAER-MoE':
                            laer_moe_data = data
                            break
                    
                    if laer_moe_data:
                        slowdown_ratio = time / laer_moe_data['a2a']
                        ax.text(bar.get_x() + bar.get_width()/2, bottom + height/2 + 1.4,
                               f'↑{slowdown_ratio:.2f}×', ha='center', va='bottom', 
                               fontsize=5, fontweight='bold', color='white')

def plot_subplot(ax, plot_data, labels, model_name):
    """Plot single subplot"""
    if not plot_data:
        return
    
    colors = ['#1f4e79', '#5b9bd5', '#a5c6e8']
    a2a_values = [d['a2a'] for d in plot_data]
    mlp_values = [d['mlp'] for d in plot_data]
    other_values = [d['other'] for d in plot_data]
    
    x_pos = np.arange(len(labels))
    bar_width = 0.6
    
    bars3 = ax.bar(x_pos, other_values, bar_width, color=colors[2], alpha=0.9, 
                   edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x_pos, mlp_values, bar_width, color=colors[1], alpha=0.9, 
                   bottom=other_values, edgecolor='black', linewidth=0.8)
    bars1 = ax.bar(x_pos, a2a_values, bar_width, color=colors[0], alpha=0.9, 
                   bottom=[o + m for o, m in zip(other_values, mlp_values)], 
                   edgecolor='black', linewidth=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=6, rotation=0)
    
    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel('Time (seconds)', fontsize=7.5, labelpad=0)
    
    ax.text(0.5, -0.3, model_name, ha='center', va='center', 
            transform=ax.transAxes, fontsize=7.5)
    
    add_percentage_labels(ax, bars3, other_values, [0]*len(plot_data), 
                         [d['profiler_time'] for d in plot_data])
    add_percentage_labels(ax, bars2, mlp_values, other_values, 
                         [d['profiler_time'] for d in plot_data])
    add_percentage_labels(ax, bars1, a2a_values, 
                         [o + m for o, m in zip(other_values, mlp_values)], 
                         [d['profiler_time'] for d in plot_data], 
                         labels=labels, plot_data=plot_data, white=True)
    
    max_total = max([d['profiler_time'] for d in plot_data])
    ax.set_ylim(0, max_total * 1.15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.8, linestyle='-')
    
    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.tick_params(axis='y', which='major', labelsize=7.5)

def plot_time_timeline(df, output_dir=None):
    """Plot stacked bar chart showing time breakdown"""
    if df is None or df.empty:
        print("No data to plot")
        return
    
    plot_data_16, plot_data_8, labels_16, labels_8 = prepare_plot_data(df)
    
    if not plot_data_16 and not plot_data_8:
        print("No valid data for plotting")
        return
    
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
        'ytick.minor.size': 2
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.17, 1.2))
    colors = ['#1f4e79', '#5b9bd5', '#a5c6e8']
    
    plot_subplot(ax1, plot_data_8, labels_8, 'Mixtral-8x7b-e8k2')
    plot_subplot(ax2, plot_data_16, labels_16, 'Mixtral-8x7b-e16k4')
    
    legend_elements = [
        Patch(facecolor=colors[0], label='All-to-All'),
        Patch(facecolor=colors[1], label='Expert'),
        Patch(facecolor=colors[2], label='Others')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1), 
               fontsize=6, frameon=True, fancybox=False, edgecolor='none', 
               shadow=False, ncol=3, columnspacing=1.0)
    
    plt.subplots_adjust(left=0.12, right=0.92, top=0.85, bottom=0.2, wspace=0.25)
    
    chart_path = os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/figure_10a.pdf')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")

def main():
    """Main function"""
    csv_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, CSV_FILE)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("Starting to plot breakdown chart...")
    print(f"CSV file: {csv_file}")
    
    df = load_data(csv_file)
    if df is not None:
        plot_time_timeline(df, output_dir)
        print("\nPlotting completed!")
    else:
        print("Failed to load data")

if __name__ == "__main__":
    main()
