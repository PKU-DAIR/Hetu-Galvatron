import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Configuration
CSV_FILE = 'LAER-MoE/scripts_ae/e2e_analysis.csv'
BACKUP_CSV_FILE = 'LAER-MoE/scripts_ae/e2e_analysis_backup.csv'
COLORS = {
    'megatron': '#A23B72',
    'FSDP': '#2E86AB',
    'FLEX': '#F18F01',
    'LAER': '#C73E1D'
}

PATTERNS = {
    'megatron': '//',
    'FSDP': '..',
    'FLEX': 'xx',
    'LAER': '\\\\'
}

METHOD_LABELS = {
    'megatron': 'Megatron',
    'FSDP': 'FSDP+EP',
    'FLEX': 'FlexMoE',
    'LAER': 'LAER-MoE'
}

# Default complete lists for robustness
DEFAULT_MODELS = [
    'qwen-8x7b-e8k2',
    'mixtral-8x7b-e8k2',
    'mixtral-8x22b-e8k2',
    'qwen-8x7b-e16k4',
    'mixtral-8x7b-e16k4',
    'mixtral-8x22b-e16k4'
]

DEFAULT_METHODS = ['megatron', 'FSDP', 'FLEX', 'LAER']
DEFAULT_AUX_LOSS = [0, 0.0001]

def load_data(csv_file):
    """Load data from CSV file with error handling."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Validate required columns
    required_cols = ['dataset', 'method', 'model_name', 'aux_loss', 'avg_time_ms']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df

def filter_data(df, dataset, aux_loss=None):
    """Filter data by dataset and optionally by aux_loss."""
    filtered = df[df['dataset'] == dataset]
    if aux_loss is not None:
        filtered = filtered[filtered['aux_loss'] == aux_loss]
    return filtered

def get_model_names(df):
    """Get sorted model names (e8k2 before e16k4). Use default models if df is empty."""
    if df.empty:
        return DEFAULT_MODELS
    
    # Get models from data, but ensure all default models are included
    models_in_data = set(df['model_name'].unique())
    all_models = list(set(DEFAULT_MODELS) | models_in_data)
    
    return sorted(all_models, key=lambda x: ('e8k2' in x, x), reverse=True)

def simplify_model_label(model_name):
    """Create simplified model label for display."""
    if 'qwen' in model_name:
        base = 'qwen-\n8x7b'
    elif 'mixtral-8x7b' in model_name:
        base = 'mixtral-\n8x7b'
    elif 'mixtral-8x22b' in model_name:
        base = 'mixtral-\n8x22b'
    else:
        base = model_name
    
    if 'e8k2' in model_name:
        suffix = 'e8k2'
    elif 'e16k4' in model_name:
        suffix = 'e16k4'
    else:
        suffix = ''
    
    return f'{base}\n{suffix}' if suffix else base

def get_time_data(df, model_name, method, aux_loss):
    """Extract time data for a specific model, method, and aux_loss.
    Returns (0, 0) if data is missing (for robustness).
    """
    try:
        data = df[
            (df['model_name'] == model_name) &
            (df['method'] == method) &
            (df['aux_loss'] == aux_loss)
        ]
        
        if data.empty:
            return 0, 0
        
        avg_time = data['avg_time_ms'].iloc[0] / 1000  # Convert to seconds
        
        # Safely get std_time
        if 'std_time_ms' in data.columns and not pd.isna(data['std_time_ms'].iloc[0]):
            std_time = data['std_time_ms'].iloc[0] / 1000
        else:
            std_time = 0
        
        return avg_time, std_time
    except (KeyError, IndexError, ValueError):
        return 0, 0

def plot_bar(ax, x_pos, time, std, method, bar_width, offset):
    """Plot a single bar with error bars. Skip if time is 0 or None."""
    if time is None or time == 0:
        return
    
    color = COLORS.get(method, '#808080')
    pattern = PATTERNS.get(method, '')
    
    ax.bar(x_pos + offset * bar_width, time, bar_width,
           color=color, hatch=pattern, edgecolor='black',
           linewidth=0.5, yerr=std, capsize=2)

def add_speedup_labels(ax, x_base, bar_width, megatron_time, baseline_time, solver_time, offset, solver_name):
    """Add speedup labels above bars. Only show labels for non-zero baselines."""
    # Skip if solver_time is 0, None, or invalid
    if solver_time is None or solver_time == 0:
        return
    
    # Get current y-axis maximum for relative scaling
    current_ymax = ax.get_ylim()[1]
    
    # Different offsets for FLEX and LAER, using relative scaling
    if solver_name == 'FLEX':
        y_offset_1 = solver_time + current_ymax * 0.45
        y_offset_2 = solver_time + current_ymax * 0.25
    else:  # LAER
        y_offset_1 = solver_time + current_ymax * 0.24
        y_offset_2 = solver_time + current_ymax * 0.03
    
    labels_added = []
    
    # Add megatron speedup if available and non-zero
    if megatron_time is not None and megatron_time > 0:
        megatron_speedup = megatron_time / solver_time
        separator = ',' if (baseline_time is not None and baseline_time > 0) else ''
        ax.text(x_base + offset * bar_width, y_offset_1,
                f'{megatron_speedup:.2f}x{separator}',
                ha='left', va='bottom', fontsize=5, color=COLORS['megatron'],
                rotation=45, fontweight='bold')
        labels_added.append(True)
    
    # Add baseline (FSDP) speedup if available and non-zero
    if baseline_time is not None and baseline_time > 0:
        baseline_speedup = baseline_time / solver_time
        # Adjust vertical position if megatron label was not added
        y_pos = y_offset_2 if (megatron_time is not None and megatron_time > 0) else y_offset_1
        ax.text(x_base + offset * bar_width, y_pos,
                f'{baseline_speedup:.2f}x',
                ha='left', va='bottom', fontsize=5, color=COLORS['FSDP'],
                rotation=45, fontweight='bold')
        labels_added.append(True)

def create_subplot(ax, df, dataset, aux_loss, subplot_index):
    """Create a single subplot for given dataset and aux_loss."""
    data = filter_data(df, dataset, aux_loss)
    
    # Always use default models for consistent layout, even if data is empty
    model_names = DEFAULT_MODELS.copy()
    
    if not model_names:
        return
    
    x_positions = np.arange(len(model_names))
    bar_width = 0.2
    
    # Set x-axis first to ensure it's always configured, even with no data
    simplified_labels = [simplify_model_label(m) for m in model_names]
    ax.set_xlim(-0.5, len(model_names) - 0.5)  # Set x-axis range
    ax.set_xticks(x_positions)
    # Only show labels on bottom row (indices 2 and 3)
    if subplot_index in [2, 3]:
        ax.set_xticklabels(simplified_labels, fontsize=5)
    else:
        # Hide labels for top row
        ax.set_xticklabels([""] * len(model_names), fontsize=5)
    
    # Plot bars for each model
    for j, model_name in enumerate(model_names):
        x_base = j
        
        # Get time data for each method
        megatron_time, megatron_std = get_time_data(data, model_name, 'megatron', aux_loss)
        none_time, none_std = get_time_data(data, model_name, 'FSDP', aux_loss)
        flex_time, flex_std = get_time_data(data, model_name, 'FLEX', aux_loss)
        laer_time, laer_std = get_time_data(data, model_name, 'LAER', aux_loss)
        
        # Plot bars
        plot_bar(ax, x_base, megatron_time, megatron_std, 'megatron', bar_width, -1.5)
        plot_bar(ax, x_base, none_time, none_std, 'FSDP', bar_width, -0.5)
        plot_bar(ax, x_base, flex_time, flex_std, 'FLEX', bar_width, 0.5)
        plot_bar(ax, x_base, laer_time, laer_std, 'LAER', bar_width, 1.5)
        
        # Add speedup labels
        if flex_time and flex_time > 0:
            add_speedup_labels(ax, x_base, bar_width, megatron_time, none_time, flex_time, 0, 'FLEX')
        if laer_time and laer_time > 0:
            add_speedup_labels(ax, x_base, bar_width, megatron_time, none_time, laer_time, 1, 'LAER')
    
    # X-axis is already set above, just configure other properties
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis (handle case where no data is plotted)
    current_ymax = ax.get_ylim()[1]
    if current_ymax > 0:
        y_max = current_ymax * 1.5
        ax.set_ylim(0, y_max)
        
        # Set y-ticks based on range
        if y_max <= 100:
            y_ticks = np.arange(0, y_max, 15)
        elif y_max <= 200:
            y_ticks = np.arange(0, y_max, 20)
        elif y_max <= 500:
            y_ticks = np.arange(0, y_max, 50)
        else:
            y_ticks = np.arange(0, y_max, 100)
        ax.set_yticks(y_ticks)
    else:
        # No data plotted, set default range
        ax.set_ylim(0, 60)
        ax.set_yticks([0, 15, 30, 45, 60])
    
    # Set title
    aux_label = '0.0' if aux_loss == 0.0 else '1e-4'
    ax.set_title(f'{dataset}, aux_loss = {aux_label}', fontsize=7.5, pad=0)

def create_combined_chart(df):
    """Create combined performance comparison chart (2x2 subplots)."""
    # Fill missing combinations with 0 for robustness
    datasets = ['wikitext', 'C4']
    missing_rows = []
    
    for dataset in datasets:
        for model in DEFAULT_MODELS:
            for method in DEFAULT_METHODS:
                for aux_loss in DEFAULT_AUX_LOSS:
                    existing = df[
                        (df['dataset'] == dataset) &
                        (df['model_name'] == model) &
                        (df['method'] == method) &
                        (df['aux_loss'] == aux_loss)
                    ]
                    if existing.empty:
                        # Add missing row with 0 values
                        missing_rows.append({
                            'dataset': dataset,
                            'method': method,
                            'model_name': model,
                            'aux_loss': aux_loss,
                            'avg_time_ms': 0,
                            'std_time_ms': 0,
                            'matched_iterations': 0,
                            'total_iterations': 0
                        })
    
    # Add missing rows to dataframe
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        df = pd.concat([df, missing_df], ignore_index=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 2.5))
    axes = axes.flatten()
    
    subplot_configs = [
        ('wikitext', 0.0),
        ('wikitext', 0.0001),
        ('C4', 0.0),
        ('C4', 0.0001)
    ]
    
    for i, (dataset, aux_loss) in enumerate(subplot_configs):
        create_subplot(axes[i], df, dataset, aux_loss, i)
    
    # Add shared y-axis label
    fig.text(0.04, 0.5, 'Avg. Iter. Time (s)', fontsize=7.5, rotation=90,
             verticalalignment='center', horizontalalignment='center')
    
    # Create legend - only include methods that actually exist in the data
    available_methods = set(df['method'].unique())
    
    # Create legend only for available methods, maintaining order
    legend_order = ['megatron', 'FSDP', 'FLEX', 'LAER']
    legend_elements = [
        Patch(facecolor=COLORS[k], hatch=PATTERNS[k], edgecolor='black', label=METHOD_LABELS[k])
        for k in legend_order if k in available_methods
    ]
    
    if legend_elements:
        ncol = min(len(legend_elements), 4)  # Max 4 columns
        fig.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, 0.98), ncol=ncol, fontsize=7.5, frameon=True)
    
    plt.subplots_adjust(left=0.08, bottom=0.2, top=0.8, hspace=0.3, wspace=0.2)
    
    # Save figures
    pdf_file = BASE_DIR + f'/LAER-MoE/scripts_ae/figure_8.pdf'

    plt.savefig(pdf_file, bbox_inches='tight')
    
    print(f"Charts saved: {pdf_file}")

def print_data_summary(df):
    """Print summary of loaded data."""
    print("=== Data Summary ===")
    
    for dataset in df['dataset'].unique():
        print(f"\n--- {dataset.upper()} ---")
        dataset_df = filter_data(df, dataset)
        
        print(f"Records: {len(dataset_df)}")
        print(f"Models: {sorted(dataset_df['model_name'].unique())}")
        print(f"Methods: {sorted(dataset_df['method'].unique())}")
        print(f"Aux Loss values: {sorted(dataset_df['aux_loss'].unique())}")

def main(type=None):
    """Main function."""
    print("Creating combined performance comparison chart...")
    
    if type == "default":
        csv_file = os.path.join(BASE_DIR, BACKUP_CSV_FILE)
    else:
        csv_file = os.path.join(BASE_DIR, CSV_FILE)
    try:
        df = load_data(csv_file)
        print(f"Loaded {len(df)} records from {csv_file}")
        
        print_data_summary(df)
        create_combined_chart(df)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None) 
