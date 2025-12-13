#!/usr/bin/env python3
"""
Training Log Analysis Script - Plotting Part
Plot loss curves from analyzed data
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def smooth_curve(x, y, window_size=10):
    """Smooth curve using moving average."""
    if len(y) < window_size:
        return x, y
    
    kernel = np.ones(window_size) / window_size
    smoothed_y = np.convolve(y, kernel, mode='valid')
    smoothed_x = x[window_size-1:]
    
    return smoothed_x, smoothed_y
    
def load_data(csv_file):
    """Load analyzed data from CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Validate required columns
    required_cols = ['iter', 'method', 'aux_loss', 'loss', 'time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Find minimum iteration count across all method+aux_loss combinations
    min_iter_count = None
    for (method, aux_loss), group in df.groupby(['method', 'aux_loss']):
        iter_count = len(group)
        if min_iter_count is None or iter_count < min_iter_count:
            min_iter_count = iter_count
    
    # Filter data to use only the first min_iter_count iterations for each combination
    if min_iter_count is not None and min_iter_count > 0:
        filtered_dfs = []
        for (method, aux_loss), group in df.groupby(['method', 'aux_loss']):
            group_sorted = group.sort_values('iter')
            # Take only the first min_iter_count iterations
            group_filtered = group_sorted.head(min_iter_count)
            filtered_dfs.append(group_filtered)
        df = pd.concat(filtered_dfs, ignore_index=True)
        print(f"Filtered to {min_iter_count} iterations per method+aux_loss combination")
    
    # Calculate cumulative time for each method+aux_loss combination
    df['cumulative_time_h'] = 0.0
    for (method, aux_loss), group in df.groupby(['method', 'aux_loss']):
        group_sorted = group.sort_values('iter')
        group_sorted['cumulative_time_h'] = group_sorted['time'].cumsum() / (1000 * 3600)
        df.loc[group_sorted.index, 'cumulative_time_h'] = group_sorted['cumulative_time_h']
    
    return df

def create_plots(df):
    """Create loss curves plots."""
    # Set academic paper style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['font.size'] = 7.5
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.17, 1))
    
    # Colors and line styles
    colors = {
        'megatron_1e2': '#1f4e79',
        'megatron_1e4': '#5b9bd5',
        'galvatron': '#C73E1D'
    }
    
    styles = {
        'megatron_1e2': '-',
        'megatron_1e4': '--',
        'galvatron': ':'
    }
    
    legend_handles = []
    legend_labels = []
    
    # Store smoothed data for later use
    smoothed_data = {}

    # Define plotting order: megatron first, then LAER
    method_order = ['megatron', 'LAER']
    
    # Group data by method and aux_loss, then sort by method order
    grouped_data = list(df.groupby(['method', 'aux_loss']))
    
    # Sort by method order (megatron first, then LAER, then others)
    def sort_key(item):
        (method, aux_loss), _ = item
        if method in method_order:
            return (method_order.index(method), aux_loss)
        else:
            return (len(method_order), aux_loss)
    
    grouped_data_sorted = sorted(grouped_data, key=sort_key)

    # Plot in specified order
    for (method, aux_loss), group_df in grouped_data_sorted:
        group_df = group_df.sort_values('iter').copy()
        
        # Determine label and color based on method and aux_loss
        if method == 'megatron':
            if aux_loss == 0.01 or abs(aux_loss - 1e-2) < 1e-6:
                label = 'Megatron 1e-2'
                color_key = 'megatron_1e2'
            else:
                label = 'Megatron 1e-4'
                color_key = 'megatron_1e4'
        elif method == 'LAER':
            label = 'LAER-MoE 1e-4'
            color_key = 'galvatron'
        else:
            label = f'{method} {aux_loss}'
            color_key = 'megatron_1e4'
        
        color = colors.get(color_key, '#808080')
        style = styles.get(color_key, '-')
        
        # Smooth data
        iterations = group_df['iter'].values
        losses = group_df['loss'].values
        smooth_iterations, smooth_losses = smooth_curve(iterations, losses)
        
        # Map smoothed iterations to cumulative time
        time_indices = [np.argmin(np.abs(iterations - iter_val)) for iter_val in smooth_iterations]
        smooth_times = group_df['cumulative_time_h'].iloc[time_indices].values
        
        # Store smoothed data for later use
        smoothed_data[(method, aux_loss)] = {
            'smooth_iterations': smooth_iterations,
            'smooth_losses': smooth_losses,
            'smooth_times': smooth_times,
            'final_smooth_loss': smooth_losses[-1] if len(smooth_losses) > 0 else None,
            'final_smooth_time': smooth_times[-1] if len(smooth_times) > 0 else None
        }
        
        # Plot original data (light)
        ax1.plot(group_df['cumulative_time_h'], group_df['loss'], 
                color=color, alpha=0.2, linewidth=0.8)
        ax2.plot(iterations, losses, 
                color=color, alpha=0.2, linewidth=0.8)
        
        # Plot smoothed data (bold)
        ax1.plot(smooth_times, smooth_losses, 
                color=color, linewidth=1.0, alpha=0.9, linestyle=style)
        ax2.plot(smooth_iterations, smooth_losses, 
                color=color, linewidth=1.0, alpha=0.9, linestyle=style)
        
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=1.5, linestyle=style))
        legend_labels.append(label)
    
    # Calculate dynamic axis limits based on data
    # Time: use max time rounded up to nearest hour
    max_time = df['cumulative_time_h'].max() if not df.empty else 16.5
    max_time_ceil = max_time * 1.05
    
    # Iteration: use max iteration from filtered data (which is based on min_iter_count)
    max_iter = int(df['iter'].max()) if not df.empty else 3000
    
    # Loss: use min loss rounded down, upper bound is min_loss_floor + 2
    min_loss = df['loss'].min() if not df.empty else 4.0
    min_loss_floor = np.floor(min_loss)
    max_loss = min_loss_floor + 2.0
    
    # Configure axes
    ax1.set_xlabel('Cumulative Time (h)', fontsize=7.5)
    ax1.set_ylabel('Loss', fontsize=7.5)
    ax1.set_title('(a)', fontsize=7.5, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax1.set_ylim(min_loss_floor, max_loss)
    ax1.set_xlim(left=0, right=max_time_ceil)
    
    # Set x-ticks for time (every 2 hours)
    if max_time_ceil > 16:
        ax1.set_xticks([2, 4, 6, 8, 10, 12, 14, 16]) 
    else:
        x_ticks = np.linspace(0, max_time_ceil, 8)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels([f'{x:.2f}' for x in x_ticks], fontsize=5)
    
    # Set y-ticks for loss (5 ticks from min_loss_floor to max_loss)
    y_ticks = np.linspace(min_loss_floor, max_loss, 5)
    ax1.set_yticks(y_ticks)
    
    ax2.set_xlabel('Iteration', fontsize=7.5)
    ax2.set_title('(b)', fontsize=7.5, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linewidth=0.5, axis='y')
    ax2.set_xlim(left=0, right=max_iter)
    ax2.set_ylim(min_loss_floor, max_loss)
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(['', '', '', '', ''])
    
    # Check if required data exists (both Megatron 1e-4 and LAER-MoE 1e-4)
    # Note: Megatron 1e-2 is optional - plot will work without it
    megatron_1e4_smooth_data = None
    laer_smooth_data = None
    
    # Find smoothed data for Megatron 1e-4 and LAER
    for (method, aux_loss), data in smoothed_data.items():
        if method == 'megatron' and ((aux_loss == 0.0001) or (abs(aux_loss - 1e-4) < 1e-6)):
            megatron_1e4_smooth_data = data
        elif method == 'LAER':
            laer_smooth_data = data
    
    # Add gray horizontal line and annotations only if both required methods (1e-4) exist
    if megatron_1e4_smooth_data is not None and laer_smooth_data is not None:
        laer_final_smooth_loss = laer_smooth_data['final_smooth_loss']
        laer_final_smooth_time = laer_smooth_data['final_smooth_time']
        megatron_final_smooth_time = megatron_1e4_smooth_data['final_smooth_time']
        megatron_final_smooth_loss = megatron_1e4_smooth_data['final_smooth_loss']
        
        # Add gray horizontal line at LAER smoothed final loss value
        if laer_final_smooth_loss is not None:
            ax1.axhline(y=laer_final_smooth_loss, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
        
        # Manually calculate speedup: time to reach the same loss value
        # Or use final time ratio if loss values are similar
        if (megatron_final_smooth_time is not None and laer_final_smooth_time is not None and
            megatron_final_smooth_loss is not None and laer_final_smooth_loss is not None):
            speedup = megatron_final_smooth_time / laer_final_smooth_time
            
            # Add loss value label on y-axis (using LAER smoothed final loss)
            ax1.text(-max_time / 50, laer_final_smooth_loss - 0.05, f'{laer_final_smooth_loss:.2f}', 
                    ha='right', va='center', fontsize=7, color='black')
            
            # Add annotation
            arrow_y = max(laer_final_smooth_loss, megatron_final_smooth_loss)
            arrow_start = min(laer_final_smooth_time, megatron_final_smooth_time) + max_time / 80
            arrow_end = max(laer_final_smooth_time, megatron_final_smooth_time) - max_time / 80
            
            ax1.annotate('', xy=(arrow_end, arrow_y), xytext=(arrow_start, arrow_y),
                        arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
            
            arrow_mid = (arrow_start + arrow_end) / 2
            ax1.text(arrow_mid, arrow_y - 0.35, f'{speedup:.1f}x', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    else:
        # Warn if required data is missing (but still allow plotting)
        missing = []
        if megatron_1e4_smooth_data is None:
            missing.append("Megatron 1e-4")
        if laer_smooth_data is None:
            missing.append("LAER-MoE 1e-4")
        print(f"\nWarning: Missing required data for annotations: {', '.join(missing)}")
        print("  Plot will be generated without speedup annotations and gray reference line.")
        print("  Note: Megatron 1e-2 is optional and will be plotted if available.")
    
    # Add legend
    plt.subplots_adjust(top=0.80, bottom=0.15, left=0.15, right=0.95, hspace=0.3)
    fig.legend(legend_handles, legend_labels, loc='upper center', 
              bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=7.5)
    
    # Save plots
    plt.savefig(BASE_DIR + '/LAER-MoE/scripts_ae/figure_9.pdf', dpi=300, bbox_inches='tight')

    print("Plots saved: figure_9.pdf")

def main(type=None):
    """Main plotting function."""
    BACKUP_CSV_FILE = 'LAER-MoE/scripts_ae/convergence_analysis_backup.csv'
    CSV_FILE = 'LAER-MoE/scripts_ae/convergence_analysis.csv'
    if type == "default":
        csv_file = os.path.join(BASE_DIR, BACKUP_CSV_FILE)
    else:
        csv_file = os.path.join(BASE_DIR, CSV_FILE)
    df = load_data(csv_file)
    print(f"Loaded {len(df)} records from {csv_file}")
    create_plots(df)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None) 
