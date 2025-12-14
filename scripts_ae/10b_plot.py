#!/usr/bin/env python3
"""
Plot token analysis from CSV file
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_FILE = 'LAER-MoE/scripts_ae/token_analysis.csv'
BACKUP_CSV_FILE = 'LAER-MoE/scripts_ae/token_analysis_backup.csv'

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
        required_columns = ['rank', 'method', 'model_name', 'layer', 'iter', 'token_num']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        print(f"Loaded {len(df)} records from {csv_file}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def prepare_plot_data(df, config_2_iter_range=(15, 16), config_4_iter_range=(51, 52)):
    """Prepare data for plotting by calculating max values per layer"""
    plot_data = {}
    
    # Expected models and methods
    expected_models = ['mixtral-8x7b-e8k2', 'mixtral-8x7b-e16k4']
    expected_methods = ['FSDP', 'FLEX', 'LAER']
    
    # Get available models and methods from data
    available_models = df['model_name'].unique().tolist() if not df.empty else []
    available_methods = df['method'].unique().tolist() if not df.empty else []
    
    # Check for missing models
    missing_models = [m for m in expected_models if m not in available_models]
    if missing_models:
        print(f"Warning: Missing model data: {missing_models}")
    
    # Check for missing methods per model
    for model in expected_models:
        if model in available_models:
            model_data = df[df['model_name'] == model]
            model_methods = model_data['method'].unique().tolist()
            missing_methods = [m for m in expected_methods if m not in model_methods]
            if missing_methods:
                print(f"Warning: Model {model} missing methods: {missing_methods}")
    
    # Group by method and model
    for (method, model_name), group in df.groupby(['method', 'model_name']):
        if method not in plot_data:
            plot_data[method] = {'config_2': [], 'config_4': [], 'perfect_balance_2': [], 'perfect_balance_4': []}
        
        # Determine config based on model_name
        if model_name == 'mixtral-8x7b-e8k2':
            max_layers = 32
            iter_range = config_2_iter_range
            config_key = 'config_2'
            perfect_key = 'perfect_balance_2'
        elif model_name == 'mixtral-8x7b-e16k4':
            max_layers = 24
            iter_range = config_4_iter_range
            config_key = 'config_4'
            perfect_key = 'perfect_balance_4'
        else:
            continue
        
        max_values = []
        perfect_balance_values = []
        
        for layer in range(max_layers):
            layer_data = group[group['layer'] == layer]
            if layer_data.empty:
                max_values.append(0)
                perfect_balance_values.append(0)
                continue
            
            # Filter by iteration range
            iter_filtered = layer_data[
                (layer_data['iter'] >= iter_range[0]) & 
                (layer_data['iter'] <= iter_range[1])
            ]
            
            if iter_filtered.empty:
                max_values.append(0)
                perfect_balance_values.append(0)
                continue
            
            # Calculate max token per iteration, then average
            layer_max_values = []
            layer_perfect_values = []
            
            for iter_idx in range(iter_range[0], iter_range[1] + 1):
                iter_data = iter_filtered[iter_filtered['iter'] == iter_idx]
                if not iter_data.empty:
                    max_token = iter_data['token_num'].max()
                    layer_max_values.append(max_token)
                    total_tokens = iter_data['token_num'].sum()
                    perfect_balance = total_tokens / 32
                    layer_perfect_values.append(perfect_balance)
            
            if layer_max_values:
                max_values.append(np.mean(layer_max_values))
                perfect_balance_values.append(np.mean(layer_perfect_values))
            else:
                max_values.append(0)
                perfect_balance_values.append(0)
        
        plot_data[method][config_key] = max_values
        plot_data[method][perfect_key] = perfect_balance_values
    
    return plot_data

def plot_subplot(ax, plot_data, solver_order, layers, model_name, config_key, perfect_key, global_max):
    """Plot single subplot"""
    colors = ['#2E86AB', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^']
    legend_handles = []
    
    # Plot solver lines
    for idx, solver_type in enumerate(solver_order):
        if solver_type in plot_data and plot_data[solver_type][config_key] and any(v > 0 for v in plot_data[solver_type][config_key]):
            data = plot_data[solver_type][config_key]
            normalized_data = [v / global_max if global_max > 0 else 0 for v in data]
            line = ax.plot(layers, normalized_data,
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    label=solver_type,
                    markerfacecolor='white',
                    markeredgewidth=1.0,
                    linewidth=1.0,
                    markersize=1)
            legend_handles.extend(line)
    
    # Add perfect balance line (use first available method's data)
    perfect_balance_data = None
    for solver_type in solver_order:
        if solver_type in plot_data and plot_data[solver_type][perfect_key] and any(v > 0 for v in plot_data[solver_type][perfect_key]):
            perfect_balance_data = plot_data[solver_type][perfect_key]
            break
    
    if perfect_balance_data:
        perfect_normalized = [v / global_max if global_max > 0 else 0 for v in perfect_balance_data]
        perfect_line = ax.plot(layers, perfect_normalized,
                color='grey', linestyle='--', linewidth=1.0,
                alpha=0.8, label='Perfect Balance')
        legend_handles.extend(perfect_line)
    
    return legend_handles

def plot_token_analysis(df, config_2_iter_range=(15, 16), config_4_iter_range=(51, 52)):
    """Plot routing token analysis"""
    if df is None or df.empty:
        print("Error: No data to plot")
        return
    
    plot_data = prepare_plot_data(df, config_2_iter_range, config_4_iter_range)
    
    if not plot_data:
        print("Error: No valid data for plotting")
        return
    
    solver_order = [f'FSDP', 'FLEX', 'LAER']
    
    # Calculate global max for normalization
    global_max_2 = max([max(data['config_2']) for method, data in plot_data.items() 
                        if data['config_2'] and any(v > 0 for v in data['config_2'])], default=1)
    global_max_4 = max([max(data['config_4']) for method, data in plot_data.items() 
                        if data['config_4'] and any(v > 0 for v in data['config_4'])], default=1)
    
    # Calculate perfect balance normalized values for y-axis limits
    perfect_balance_2_normalized = None
    perfect_balance_4_normalized = None
    
    for solver_type in solver_order:
        if solver_type in plot_data:
            # For config_2
            if plot_data[solver_type]['perfect_balance_2'] and any(v > 0 for v in plot_data[solver_type]['perfect_balance_2']):
                perfect_balance_2_normalized = [v / global_max_2 if global_max_2 > 0 else 0 
                                                for v in plot_data[solver_type]['perfect_balance_2']]
                break
    
    for solver_type in solver_order:
        if solver_type in plot_data:
            # For config_4
            if plot_data[solver_type]['perfect_balance_4'] and any(v > 0 for v in plot_data[solver_type]['perfect_balance_4']):
                perfect_balance_4_normalized = [v / global_max_4 if global_max_4 > 0 else 0 
                                                for v in plot_data[solver_type]['perfect_balance_4']]
                break
    
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
    
    # Plot config 2 (e8k2, 32 layers)
    layers_2 = list(range(32))
    legend_handles_2 = plot_subplot(ax1, plot_data, solver_order, layers_2, 
                                    'Mixtral-8x7B e8k2', 'config_2', 'perfect_balance_2', global_max_2)
    
    ax1.set_title('Mixtral-8x7B e8k2', fontsize=7.5, pad=0)
    ax1.set_xlabel('Layer Index', fontsize=7.5)
    try:
        if ax1.get_subplotspec().colspan.start == 0:
            ax1.set_ylabel('Relative Max Token Count', fontsize=7.5)
    except (AttributeError, ValueError):
        ax1.set_ylabel('Relative Max Token Count', fontsize=7.5)
    ax1.set_xlim(-0.5, 31.5)
    # Calculate y-axis lower limit based on perfect balance value
    if perfect_balance_2_normalized:
        perfect_mean_2 = np.mean([v for v in perfect_balance_2_normalized if v > 0]) if any(v > 0 for v in perfect_balance_2_normalized) else 0.3
        y_min_2 = max(0, perfect_mean_2 - 0.1)
    else:
        y_min_2 = 0.3
    ax1.set_ylim(y_min_2, 1.05)
    ax1.set_xticks(range(0, 32, 4))
    ax1.set_yticks(np.arange(0.4, 1.05, 0.2))
    ax1.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    ax1.tick_params(axis='both', which='major', labelsize=7.5)
    
    # Plot config 4 (e16k4, 24 layers)
    layers_4 = list(range(24))
    legend_handles_4 = plot_subplot(ax2, plot_data, solver_order, layers_4,
                                    'Mixtral-8x7B e16k4', 'config_4', 'perfect_balance_4', global_max_4)
    
    ax2.set_title('Mixtral-8x7B e16k4', fontsize=7.5, pad=0)
    ax2.set_xlabel('Layer Index', fontsize=7.5)
    ax2.set_xlim(-0.5, 23.5)
    # Calculate y-axis lower limit based on perfect balance value
    if perfect_balance_4_normalized:
        perfect_mean_4 = np.mean([v for v in perfect_balance_4_normalized if v > 0]) if any(v > 0 for v in perfect_balance_4_normalized) else 0.2
        y_min_4 = max(0, perfect_mean_4 - 0.1)
    else:
        y_min_4 = 0.2
    ax2.set_ylim(y_min_4, 1.05)
    ax2.set_xticks(range(0, 24, 4))
    ax2.set_yticks(np.arange(0.2, 1.05, 0.2))
    ax2.grid(True, alpha=0.3, linewidth=0.8, linestyle='-')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    ax2.tick_params(axis='both', which='major', labelsize=7.5)
    
    # Create legend
    legend_name_mapping = {'FSDP': 'FSDP+EP', 'FLEX': 'FlexMoE', 'LAER': 'LAER-MoE'}
    solver_handles = [h for h in legend_handles_2 if h.get_label() != 'Perfect Balance']
    perfect_handle = [h for h in legend_handles_2 if h.get_label() == 'Perfect Balance']
    
    solver_labels = [legend_name_mapping.get(h.get_label(), h.get_label()) 
                     for h in solver_handles if h.get_label() in legend_name_mapping]
    
    if solver_handles:
        legend1 = fig.legend(solver_handles, solver_labels,
                            loc='upper center', bbox_to_anchor=(0.5, 1.0),
                            fontsize=6, frameon=False, fancybox=False,
                            shadow=False, ncol=len(solver_labels), columnspacing=1.0)
        fig.add_artist(legend1)
    
    if perfect_handle:
        fig.legend(perfect_handle, ['Perfect Balance'],
                  loc='upper center', bbox_to_anchor=(0.5, 0.9),
                  fontsize=6, frameon=False, fancybox=False,
                  shadow=False, ncol=1, columnspacing=1.0)
    
    plt.subplots_adjust(left=0.1, right=0.92, top=0.7, bottom=0.15, wspace=0.25)
    
    chart_path = os.path.join(BASE_DIR, 'LAER-MoE/scripts_ae/figure_10b.pdf')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {chart_path}")

def main(type=None):
    """Main function"""
    if type == "default":
        csv_file = os.path.join(BASE_DIR, BACKUP_CSV_FILE)
    else:
        csv_file = os.path.join(BASE_DIR, CSV_FILE)
    
    # Default iteration ranges
    if type == "default":
        config_2_iter_range = (5, 9)
        config_4_iter_range = (25, 29)
    else:
        config_2_iter_range = (20, 39)
        config_4_iter_range = (20, 39)
    
    print("Starting to plot token analysis...")
    print(f"CSV file: {csv_file}")
    print(f"Config 2 iter range: {config_2_iter_range}")
    print(f"Config 4 iter range: {config_4_iter_range}")
    
    df = load_data(csv_file)
    if df is not None:
        plot_token_analysis(df, config_2_iter_range, config_4_iter_range)
        print("\nPlotting completed!")
    else:
        print("Failed to load data")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
