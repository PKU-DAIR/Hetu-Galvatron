#!/usr/bin/env python3
"""
Plot ablation analysis from CSV file
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_FILE = os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/ablation_analysis.csv')
BACKUP_CSV_FILE = os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/ablation_analysis_backup.csv')

def load_data(csv_file):
    """Load data from CSV file"""
    if not os.path.exists(csv_file):
        if os.path.exists(BACKUP_CSV_FILE):
            csv_file = BACKUP_CSV_FILE
            print(f"Using backup CSV: {BACKUP_CSV_FILE}")
        else:
            print(f"Error: CSV file {csv_file} not found")
            return None
    
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['method', 'avg_time_ms']
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
    """Prepare data for plotting"""
    results = []
    for _, row in df.iterrows():
        results.append({
            'method': row['method'],
            'avg_time': row['avg_time_ms']
        })
    
    return results

def get_method_label(method):
    """Get label for method"""
    method_lower = method.lower()
    
    if 'fsdp' in method_lower or 'none' in method_lower:
        return 'FSDP+EP'
    elif method_lower == 'laer' or 'laer' in method_lower:
        return 'LAER-MoE'
    elif 'no_comm_opt' in method_lower or 'no comm opt' in method_lower or 'w/o comm opt' in method_lower:
        return 'w/o comm opt'
    elif 'no_even' in method_lower or 'no even' in method_lower:
        return 'w/ pq w/o even'
    elif 'no_pq' in method_lower or 'no pq' in method_lower:
        return 'w/ even w/o pq'

def plot_results(results):
    """Plot bar chart"""
    if not results:
        print("No data to plot")
        return
    
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams.update({
        'font.size': 7.5,
        'axes.linewidth': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8
    })
    
    methods = [r['method'] for r in results]
    times = [r['avg_time'] for r in results]
    
    # Calculate relative times with robustness
    if not times or all(t == 0 or pd.isna(t) for t in times):
        print("Warning: No valid time data found")
        return
    
    # Filter out zero and NaN values for min calculation
    valid_times = [t for t in times if t > 0 and not pd.isna(t)]
    if not valid_times:
        print("Warning: No valid positive time values found")
        return
    
    min_time = min(valid_times)
    relative_times = []
    for t in times:
        if t > 0 and not pd.isna(t):
            relative_times.append(t / min_time)
        else:
            relative_times.append(0.0)  # No result treated as 0
    
    # Define color palette
    color_palette = {
        'LAER-MoE': '#C73E1D',
        'similar1': '#B8860B',
        'similar2': '#CD853F',
        'w/o comm opt': '#D2691E'
    }
    
    # Assign colors and filter baseline
    filtered_data = []
    fsdp_time = None
    
    for method, time, rel_time in zip(methods, times, relative_times):
        method_lower = method.lower()
        
        # Check if FSDP baseline
        if 'fsdp' in method_lower or 'none' in method_lower:
            fsdp_time = rel_time if rel_time > 0 else None
            continue
        
        # Get label
        label = get_method_label(method)
        
        # Determine color based on label
        if label == 'LAER-MoE':
            color = color_palette['LAER-MoE']
        elif label == 'w/o comm opt':
            color = color_palette['w/o comm opt']
        elif label == 'w/ pq w/o even':
            color = color_palette['similar1']
        elif label == 'w/ even w/o pq':
            color = color_palette['similar2']
        else:
            # Default color for unknown methods
            color = color_palette['similar1']
        
        filtered_data.append({
            'method': method,
            'label': label,
            'relative_time': rel_time,
            'original_time': time,
            'color': color
        })
    
    if not filtered_data:
        print("No data to plot after filtering")
        return
    
    # Sort by specified order: LAER, no even, no pq, no comm
    label_order = {
        'LAER-MoE': 1,
        'w/ pq w/o even': 2,
        'w/ even w/o pq': 3,
        'w/o comm opt': 4
    }
    
    def get_sort_key(item):
        label = item['label']
        return label_order.get(label, 999)  # Unknown labels go to the end
    
    filtered_data.sort(key=get_sort_key)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.17, 0.7))
    
    # Plot bars
    x_positions = range(len(filtered_data))
    heights = [d['relative_time'] for d in filtered_data]
    colors_list = [d['color'] for d in filtered_data]
    
    bars = ax.bar(x_positions, heights, 
                  color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=0.5)
    
    # Add value labels
    valid_heights_for_label = [h for h in heights if h > 0]
    max_height_for_label = max(valid_heights_for_label) if valid_heights_for_label else 1.0
    
    for bar, data in zip(bars, filtered_data):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + max_height_for_label*0.01,
                   f'{data["relative_time"]:.2f}x', ha='center', va='bottom', 
                   fontweight='bold', fontsize=7.5)
    
    # Add baseline line
    if fsdp_time:
        ax.axhline(y=fsdp_time, color='#2E86AB', linewidth=1.0, 
                  linestyle='--', alpha=0.8, label='FSDP+EP Baseline')
    
    # Set axes
    ax.set_ylabel('Relative Time', fontsize=7.5, fontweight='normal', labelpad=0)
    
    # Calculate max relative time with robustness
    valid_heights = [h for h in heights if h > 0]
    if valid_heights:
        max_relative_time = max(valid_heights)
        if fsdp_time:
            max_relative_time = max(max_relative_time, fsdp_time)
        ax.set_ylim(bottom=0.8, top=max_relative_time + 0.1)
    else:
        # All heights are 0, set a default range
        ax.set_ylim(bottom=0, top=1.2)
    
    ax.set_xticks([])
    
    # Create legend
    legend_elements = []
    for data in filtered_data:
        legend_elements.append(Patch(facecolor=data['color'], 
                                     edgecolor='black', 
                                     linewidth=0.5, 
                                     label=data['label']))
    
    if fsdp_time:
        legend_elements.append(Line2D([0], [0], color='#2E86AB', linewidth=1, 
                                     linestyle='--', alpha=0.8, label='FSDP+EP'))
    
    ax.legend(handles=legend_elements, loc='center left', 
             bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=6, frameon=False)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.subplots_adjust(right=0.6)
    
    # Save figure
    output_path = os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/figure_12.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Chart saved to: {output_path}")
    print(f"Min time: {min_time:.1f}ms (set as 1.0x)")

def main(type=None):
    """Main function"""
    if type == "default":
        csv_file = BACKUP_CSV_FILE
    else:
        csv_file = CSV_FILE
    
    df = load_data(csv_file)
    if df is None:
        print("Failed to load data")
        return
    
    results = prepare_plot_data(df)
    plot_results(results)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
