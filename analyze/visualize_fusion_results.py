#!/usr/bin/env python3
# -*- coding: gbk -*-
"""
Visualize fusion experiment results with emphasis on desired vs undesired effects.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse

def create_fusion_results_data():
    """Create the fusion results data from the table."""
    
    # Data from the table
    data = {
        'Method': [
            'Baseline',
            'Fusion: inventive+outgoing',
            'Fusion: careless+selfinterested', 
            'Fusion: calm+outgoing',
            'Subtraction: inventive-consistent',
            'Subtraction: outgoing-solitary',
            'Subtraction: dependable-selfinterested',
            'Subtraction: inventive-outgoing',
            'Subtraction: nervous-compassionate'
        ],
        'Extraversion': [3.25, 4.38, 3.81, 4.34, 4.25, 4.38, 3.79, 3.06, 2.44],
        'Agreeableness': [4.48, 4.3, 3.56, 4.26, 4.28, 4.32, 4.22, 4.21, 3.07],
        'Conscientiousness': [4.74, 3.47, 1.94, 4.44, 4.42, 2.83, 4.72, 4.44, 3.26],
        'Neuroticism': [1.68, 1.92, 3.02, 1.48, 1.7, 1.92, 1.57, 1.92, 4.01],
        'Openness': [3.91, 4.11, 3.40, 4.09, 4.17, 3.79, 3.84, 4.1, 3.37]
    }
    
    # Define desired effects for each experiment
    desired_effects = {
        'Fusion: inventive+outgoing': ['Extraversion', 'Openness'],  # Should increase both
        'Fusion: careless+selfinterested': [],  # Neutralization - no specific desired direction
        'Fusion: calm+outgoing': ['Extraversion'],  # Should increase extraversion, decrease neuroticism
        'Subtraction: inventive-consistent': ['Openness'],  # Should amplify openness
        'Subtraction: outgoing-solitary': ['Extraversion'],  # Should amplify extraversion
        'Subtraction: dependable-selfinterested': ['Agreeableness', 'Conscientiousness'],  # Should amplify both
        'Subtraction: inventive-outgoing': ['Openness'],  # Should maintain openness, suppress extraversion
        'Subtraction: nervous-compassionate': ['Neuroticism']  # Should increase neuroticism
    }
    
    # Define which changes should be suppressed/decreased
    suppressed_effects = {
        'Fusion: calm+outgoing': ['Neuroticism'],  # Should decrease neuroticism
        'Subtraction: inventive-outgoing': ['Extraversion'],  # Should suppress extraversion
        'Subtraction: nervous-compassionate': ['Agreeableness']  # Should suppress agreeableness
    }
    
    return pd.DataFrame(data), desired_effects, suppressed_effects

def calculate_effect_sizes(df, baseline_idx=0):
    """Calculate effect sizes compared to baseline."""
    baseline = df.iloc[baseline_idx, 1:].values
    effect_sizes = df.iloc[:, 1:].copy()
    
    for i in range(len(df)):
        if i != baseline_idx:
            effect_sizes.iloc[i] = df.iloc[i, 1:].values - baseline
        else:
            effect_sizes.iloc[i] = 0
    
    return effect_sizes

def create_highlighted_heatmap(df, desired_effects, suppressed_effects, save_path=None):
    """Create a heatmap that highlights desired effects and de-emphasizes undesired ones."""
    
    # Calculate effect sizes
    effect_sizes = calculate_effect_sizes(df)
    
    # Create the base heatmap data
    methods = df['Method'].values[1:]  # Exclude baseline
    dimensions = df.columns[1:].values
    
    # Create effect size matrix (excluding baseline)
    effect_matrix = effect_sizes.iloc[1:].values
    
    # Create significance masks
    desired_mask = np.zeros_like(effect_matrix, dtype=bool)
    suppressed_mask = np.zeros_like(effect_matrix, dtype=bool)
    neutral_mask = np.zeros_like(effect_matrix, dtype=bool)
    
    for i, method in enumerate(methods):
        for j, dim in enumerate(dimensions):
            if method in desired_effects and dim in desired_effects[method]:
                desired_mask[i, j] = True
            elif method in suppressed_effects and dim in suppressed_effects[method]:
                suppressed_mask[i, j] = True
            else:
                neutral_mask[i, j] = True
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left plot: Highlighted desired effects
    ax1.set_title('Fusion Experiments: Desired Effects Highlighted', fontsize=16, fontweight='bold', pad=20)
    
    # Create custom colormap for desired effects
    desired_cmap = LinearSegmentedColormap.from_list('desired', ['white', 'lightgreen', 'darkgreen'])
    suppressed_cmap = LinearSegmentedColormap.from_list('suppressed', ['darkred', 'lightcoral', 'white'])
    neutral_cmap = LinearSegmentedColormap.from_list('neutral', ['lightgray', 'lightgray'])
    
    # Plot the heatmap with different styles for different effect types
    im1 = ax1.imshow(effect_matrix, cmap='RdBu_r', vmin=-2, vmax=2, alpha=0.3)
    
    # Overlay desired effects with strong highlighting
    desired_data = effect_matrix.copy()
    desired_data[~desired_mask] = np.nan
    im_desired = ax1.imshow(desired_data, cmap=desired_cmap, vmin=0, vmax=2, alpha=0.9)
    
    # Overlay suppressed effects (where decrease is desired)
    suppressed_data = -effect_matrix.copy()  # Flip sign so decreases show as positive
    suppressed_data[~suppressed_mask] = np.nan
    im_suppressed = ax1.imshow(suppressed_data, cmap=desired_cmap, vmin=0, vmax=2, alpha=0.9)
    
    # Add text annotations with emphasis
    for i in range(len(methods)):
        for j in range(len(dimensions)):
            value = effect_matrix[i, j]
            
            if desired_mask[i, j] and value > 0.1:  # Desired increase
                ax1.text(j, i, f'{value:+.2f}', ha='center', va='center', 
                        fontweight='bold', fontsize=12, color='darkgreen',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            elif suppressed_mask[i, j] and value < -0.1:  # Desired decrease
                ax1.text(j, i, f'{value:+.2f}', ha='center', va='center', 
                        fontweight='bold', fontsize=12, color='darkgreen',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            else:  # Undesired or neutral effects
                ax1.text(j, i, f'{value:+.2f}', ha='center', va='center', 
                        fontweight='normal', fontsize=10, color='gray', alpha=0.6)
    
    # Customize axes
    ax1.set_xticks(range(len(dimensions)))
    ax1.set_xticklabels(dimensions, rotation=45, ha='right')
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels([method.replace('Fusion: ', 'F: ').replace('Subtraction: ', 'S: ') for method in methods])
    
    # Right plot: Success rate analysis
    ax2.set_title('Effect Success Analysis', fontsize=16, fontweight='bold', pad=20)
    
    # Calculate success metrics
    success_data = []
    for i, method in enumerate(methods):
        total_desired = len(desired_effects.get(method, [])) + len(suppressed_effects.get(method, []))
        successful = 0
        
        for dim in desired_effects.get(method, []):
            j = list(dimensions).index(dim)
            if effect_matrix[i, j] > 0.1:  # Threshold for meaningful increase
                successful += 1
        
        for dim in suppressed_effects.get(method, []):
            j = list(dimensions).index(dim)
            if effect_matrix[i, j] < -0.1:  # Threshold for meaningful decrease
                successful += 1
        
        success_rate = successful / total_desired if total_desired > 0 else 0
        
        # Count undesired side effects
        side_effects = 0
        total_neutral = 5 - total_desired  # Total dimensions minus desired ones
        
        for j, dim in enumerate(dimensions):
            if (method not in desired_effects or dim not in desired_effects[method]) and \
               (method not in suppressed_effects or dim not in suppressed_effects[method]):
                if abs(effect_matrix[i, j]) > 0.2:  # Significant undesired change
                    side_effects += 1
        
        side_effect_rate = side_effects / total_neutral if total_neutral > 0 else 0
        
        success_data.append({
            'Method': method.replace('Fusion: ', 'F: ').replace('Subtraction: ', 'S: '),
            'Success Rate': success_rate,
            'Side Effect Rate': side_effect_rate,
            'Net Effectiveness': success_rate - side_effect_rate
        })
    
    success_df = pd.DataFrame(success_data)
    
    # Create bar plot
    x = np.arange(len(success_df))
    width = 0.25
    
    bars1 = ax2.bar(x - width, success_df['Success Rate'], width, label='Success Rate', 
                    color='green', alpha=0.7)
    bars2 = ax2.bar(x, success_df['Side Effect Rate'], width, label='Side Effect Rate', 
                    color='red', alpha=0.7)
    bars3 = ax2.bar(x + width, success_df['Net Effectiveness'], width, label='Net Effectiveness', 
                    color='blue', alpha=0.7)
    
    ax2.set_ylabel('Rate')
    ax2.set_ylim(-0.5, 1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(success_df['Method'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add horizontal line at 0
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig

def create_selective_table_visualization(df, desired_effects, suppressed_effects, save_path=None):
    """Create a table visualization that emphasizes desired effects."""
    
    effect_sizes = calculate_effect_sizes(df)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for the table
    methods = df['Method'].values[1:]  # Exclude baseline
    dimensions = df.columns[1:].values
    
    # Create table data with formatting
    table_data = []
    for i, method in enumerate(methods):
        row = [method.replace('Fusion: ', 'F: ').replace('Subtraction: ', 'S: ')]
        
        for j, dim in enumerate(dimensions):
            value = effect_sizes.iloc[i+1, j]  # +1 because we excluded baseline
            
            # Format based on whether this is a desired effect
            if method in desired_effects and dim in desired_effects[method] and value > 0.1:
                row.append(f'↗ {value:+.2f}')  # Desired increase
            elif method in suppressed_effects and dim in suppressed_effects[method] and value < -0.1:
                row.append(f'↘ {value:+.2f}')  # Desired decrease
            elif abs(value) < 0.1:
                row.append(f'→ {value:+.2f}')  # Stable (minimal change)
            else:
                row.append(f'{value:+.2f}')  # Undesired change
        
        table_data.append(row)
    
    # Create the table
    table = ax.table(cellText=table_data,
                    colLabels=['Method'] + list(dimensions),
                    cellLoc='center',
                    loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color cells based on effect type
    for i, method in enumerate(methods):
        for j, dim in enumerate(dimensions):
            value = effect_sizes.iloc[i+1, j]
            cell = table[(i+1, j+1)]  # +1 for header row and method column
            
            if method in desired_effects and dim in desired_effects[method] and value > 0.1:
                cell.set_facecolor('#90EE90')  # Light green for desired increase
                cell.set_text_props(weight='bold')
            elif method in suppressed_effects and dim in suppressed_effects[method] and value < -0.1:
                cell.set_facecolor('#90EE90')  # Light green for desired decrease
                cell.set_text_props(weight='bold')
            elif abs(value) < 0.1:
                cell.set_facecolor('#F0F0F0')  # Light gray for stable
            else:
                cell.set_facecolor('#FFE4E1')  # Light red for undesired changes
                cell.set_alpha(0.5)
    
    # Style header
    for j in range(len(dimensions) + 1):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Fusion Experiments: Effect Size Table\n(Green = Desired Effects, Gray = Stable, Red = Side Effects)', 
              fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Table visualization saved to: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize fusion experiment results")
    parser.add_argument("--output_dir", default="analyze/fusion_analysis", help="Output directory")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", help="Output format")
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df, desired_effects, suppressed_effects = create_fusion_results_data()
    
    print("Creating fusion results visualizations...")
    
    # Create heatmap visualization
    heatmap_path = os.path.join(args.output_dir, f"fusion_effects_heatmap.{args.format}")
    fig1 = create_highlighted_heatmap(df, desired_effects, suppressed_effects, heatmap_path)
    
    # Create table visualization
    table_path = os.path.join(args.output_dir, f"fusion_effects_table.{args.format}")
    fig2 = create_selective_table_visualization(df, desired_effects, suppressed_effects, table_path)
    
    print("Visualizations completed!")
    print(f"Files saved in: {args.output_dir}")
    
    # Show plots if running interactively
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()
