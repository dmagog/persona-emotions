#!/usr/bin/env python3
# -*- coding: gbk -*-
"""
Create clean, Anthropic-style visualization focusing on target effects only.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os

# Anthropic color scheme
ANTHROPIC_COLORS = {
    'primary': '#FF6B35',      # Anthropic orange
    'orange': '#F1BF6B',      # Light orange
    'secondary': '#1E3A8A',    # Deep blue
    'success': '#22C55E',      # Green
    'warning': '#F59E0B',      # Amber
    'error': '#EF4444',        # Red
    'neutral': '#6B7280',      # Gray
    'text': '#1F2937',         # Dark gray
    'light_gray': '#F3F4F6',   # Light gray
    'success_bg': '#DCFCE7',   # Light green background
    'neutral_bg': '#F9FAFB',   # Very light gray background
}

def create_fusion_data():
    """Create the fusion results data."""
    
    data = {
        'Method': [
            'Baseline',
            'Fusion: inventive+outgoing',
            'Fusion: careless+selfinterested', 
            'Fusion: calm+outgoing',
            'Subtraction: inventive-consistent',
            'Subtraction: outgoing-solitary',
            'Subtraction: calm-nervous',
            'Subtraction: inventive-outgoing',
            'Subtraction: nervous-compassionate'
        ],
        'Extraversion': [3.25, 4.38, 3.81, 4.34, 4.25, 4.38, 3.76, 3.06, 2.44],
        'Agreeableness': [4.48, 4.3, 3.56, 4.26, 4.28, 4.32, 4.21, 4.21, 3.07],
        'Conscientiousness': [4.44, 3.47, 1.94, 4.44, 4.42, 2.83, 4.44, 4.44, 3.26],
        'Neuroticism': [1.68, 1.92, 3.02, 1.48, 1.7, 1.92, 1.39, 1.92, 4.01],
        'Openness': [3.91, 4.11, 3.40, 4.09, 4.17, 3.79, 3.74, 4.1, 3.37]
    }
    
    # Define target outcomes for each experiment
    targets = {
        'Fusion: inventive+outgoing': [
            ('Extraversion', 'increase', 'Combine creativity with social engagement'),
            ('Openness', 'increase', 'Enhance creative openness')
        ],
        'Fusion: calm+outgoing': [
            ('Extraversion', 'increase', 'Social engagement with emotional stability'),
            ('Neuroticism', 'decrease', 'Maintain calmness')
        ],
        'Subtraction: inventive-consistent': [
            ('Openness', 'increase', 'Amplify creative thinking')
        ],
        'Subtraction: outgoing-solitary': [
            ('Extraversion', 'increase', 'Enhance social engagement')
        ],
        'Subtraction: calm-nervous': [
            ('Neuroticism', 'decrease', 'Maintain calmness'),
        ],
        'Subtraction: inventive-outgoing': [
            ('Openness', 'increase', 'Creative but introverted profile'),
            ('Extraversion', 'decrease', 'Reduce social seeking')
        ],
        'Subtraction: nervous-compassionate': [
            ('Neuroticism', 'increase', 'Emotional sensitivity'),
            ('Agreeableness', 'decrease', 'Reduced prosocial focus')
        ]
    }
    
    return pd.DataFrame(data), targets

def create_anthropic_style_plot():
    """Set up Anthropic-style plot aesthetics."""
    plt.style.use('default')
    
    # Set global font and styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.edgecolor': ANTHROPIC_COLORS['neutral'],
        'axes.facecolor': '#faf9f5',
        'figure.facecolor': '#faf9f5',
        'grid.color': '#E5E7EB',
        'grid.linewidth': 0.5,
        'text.color': ANTHROPIC_COLORS['text'],
        'axes.labelcolor': ANTHROPIC_COLORS['text'],
        'xtick.color': ANTHROPIC_COLORS['text'],
        'ytick.color': ANTHROPIC_COLORS['text']
    })

def create_target_focused_visualization(df, targets, save_path=None):
    """Create a clean visualization focusing only on target effects with actual values."""
    
    create_anthropic_style_plot()
    
    # Calculate baseline and effects
    baseline = df.iloc[0, 1:].values
    methods = [m for m in df['Method'].values[1:] if m in targets]
    dimensions = df.columns[1:].values
    
    # Prepare data for visualization
    viz_data = []
    
    for method in methods:
        # Get method data
        method_row = df[df['Method'] == method].iloc[0]
        method_values = method_row.iloc[1:].values
        effects = method_values - baseline
        
        # Extract target effects only
        for dim, direction, description in targets[method]:
            dim_idx = list(dimensions).index(dim)
            effect = effects[dim_idx]
            baseline_val = baseline[dim_idx]
            new_val = method_values[dim_idx]
            
            # Determine success and styling
            if direction == 'increase':
                success = effect > 0.1
                arrow = '↗' if success else '→'
                expected_sign = '+'
            else:  # decrease
                success = effect < -0.1
                arrow = '↘' if success else '→'
                expected_sign = '-'
            
            viz_data.append({
                'method': method.replace('Fusion: ', '').replace('Subtraction: ', ''),
                'dimension': dim,
                'effect': effect,
                'baseline': baseline_val,
                'new_value': new_val,
                'success': success,
                'direction': direction,
                'arrow': arrow,
                'description': description
            })
    
    # Create the main visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Main plot: Effect magnitudes with baseline and new values
    y_positions = range(len(viz_data))
    
    for i, data in enumerate(viz_data):
        effect = data['effect']
        baseline_val = data['baseline']
        new_val = data['new_value']
        success = data['success']
        
        # Color based on success
        color = ANTHROPIC_COLORS['success'] if data['direction'] == "increase" else ANTHROPIC_COLORS['error']

        # Plot baseline value as light gray bar
        ax.barh(i, baseline_val, height=0.4, color='lightgray', alpha=0.3, label='Baseline' if i == 0 else "")
        
        # Plot new value as colored bar
        ax.barh(i, new_val, height=0.6, color=color, alpha=0.8)
        
        # Add value labels
        ax.text(max(baseline_val, new_val) + 0.1, i, 
                f'{new_val:.2f} ({effect:+.2f})', 
                va='center', fontweight='bold', color=color, fontsize=11)
        
        # Add baseline value label
        ax.text(baseline_val/2, i, f'{baseline_val:.2f}', 
                va='center', ha='center', fontsize=11)
    
    # Customize plot
    method_labels = [f"{d['method']}\n{d['dimension']} {d['arrow']}" for d in viz_data]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(method_labels, fontsize=14)
    ax.set_xlabel('Trait Score (1-5 scale)', fontweight='bold')
    # ax.set_title('Target Trait Changes', fontsize=14, fontweight='bold', 
    #               color=ANTHROPIC_COLORS['text'])
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlim(0, 6)
    ax.invert_yaxis()
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Overall title
    # fig.suptitle('Persona Vector Fusion: Target Dimension Results', 
    #             fontsize=16, fontweight='bold', color=ANTHROPIC_COLORS['text'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='#faf9f5', edgecolor='none')
        print(f"Target-focused visualization saved to: {save_path}")
    
    return fig

def create_compact_results_table(df, targets, save_path=None):
    """Create a compact table showing only target dimension results."""
    
    create_anthropic_style_plot()
    
    # Calculate baseline and effects
    baseline = df.iloc[0, 1:].values
    methods = [m for m in df['Method'].values[1:] if m in targets]
    dimensions = df.columns[1:].values
    
    # Prepare summary data
    summary_data = []
    
    for method in methods:
        # Get method data
        method_row = df[df['Method'] == method].iloc[0]
        method_values = method_row.iloc[1:].values
        effects = method_values - baseline
        
        # Extract target effects only
        for dim, direction, description in targets[method]:
            dim_idx = list(dimensions).index(dim)
            effect = effects[dim_idx]
            baseline_val = baseline[dim_idx]
            new_val = method_values[dim_idx]
            
            # Determine success
            if direction == 'increase':
                success = effect > 0.1
                target_symbol = '↗'
            else:
                success = effect < -0.1
                target_symbol = '↘'
            
            summary_data.append({
                'Experiment': method.replace('Fusion: ', '').replace('Subtraction: ', ''),
                'Target Dimension': dim,
                'Goal': target_symbol,
                'Baseline': f'{baseline_val:.2f}',
                'Result': f'{new_val:.2f}',
                'Change': f'{effect:+.2f}',
                'Success': '?' if success else '○'
            })
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table_data = []
    headers = list(summary_data[0].keys())
    
    for row in summary_data:
        table_data.append(list(row.values()))
    
    # Color rows based on success
    cell_colors = []
    for row in summary_data:
        if row['Success'] == '?':
            row_colors = [ANTHROPIC_COLORS['success_bg']] * len(headers)
        else:
            row_colors = [ANTHROPIC_COLORS['neutral_bg']] * len(headers)
        cell_colors.append(row_colors)
    
    # Create the table
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    cellColours=cell_colors)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor(ANTHROPIC_COLORS['primary'])
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.15)
    
    # Style data cells
    for i in range(1, len(summary_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            cell.set_height(0.12)
            
            # Bold the change values
            if headers[j] == 'Change':
                cell.set_text_props(weight='bold')
            
            # Color success markers
            if headers[j] == 'Success':
                if table_data[i-1][j] == '?':
                    cell.set_text_props(color=ANTHROPIC_COLORS['success'], weight='bold')
                else:
                    cell.set_text_props(color=ANTHROPIC_COLORS['neutral'])
    
    # Add title
    plt.suptitle('Fusion Experiment Results: Target Dimensions Only', 
                fontsize=16, fontweight='bold', color=ANTHROPIC_COLORS['text'], y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='#faf9f5', edgecolor='none')
        print(f"Compact results table saved to: {save_path}")
    
    return fig


def create_success_summary(df, targets, save_path=None):
    """Create a clean summary of successful target achievements."""
    
    create_anthropic_style_plot()
    
    baseline = df.iloc[0, 1:].values
    methods = [m for m in df['Method'].values[1:] if m in targets]
    dimensions = df.columns[1:].values
    
    # Calculate success rates
    success_data = []
    
    for method in methods:
        method_row = df[df['Method'] == method].iloc[0]
        method_values = method_row.iloc[1:].values
        effects = method_values - baseline
        
        total_targets = len(targets[method])
        successful_targets = 0
        
        for dim, direction, _ in targets[method]:
            dim_idx = list(dimensions).index(dim)
            actual_effect = effects[dim_idx]
            
            if direction == 'increase' and actual_effect > 0.1:
                successful_targets += 1
            elif direction == 'decrease' and actual_effect < -0.1:
                successful_targets += 1
        
        success_rate = successful_targets / total_targets if total_targets > 0 else 0
        
        success_data.append({
            'method': method.replace('Fusion: ', '').replace('Subtraction: ', ''),
            'success_rate': success_rate,
            'successful': successful_targets,
            'total': total_targets
        })
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods_short = [d['method'] for d in success_data]
    success_rates = [d['success_rate'] for d in success_data]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(methods_short)), success_rates, 
                   color=[ANTHROPIC_COLORS['success'] if sr >= 0.5 else ANTHROPIC_COLORS['primary'] 
                         for sr in success_rates],
                   alpha=0.8)
    
    # Add percentage labels
    for i, (bar, data) in enumerate(zip(bars, success_data)):
        percentage = data['success_rate'] * 100
        ax.text(data['success_rate'] + 0.02, i, f"{percentage:.0f}%",
               va='center', fontweight='bold', 
               color=ANTHROPIC_COLORS['text'])
        
        # Add success count
        ax.text(0.02, i, f"{data['successful']}/{data['total']}", 
               va='center', fontsize=9, 
               color='white' if data['success_rate'] > 0.3 else ANTHROPIC_COLORS['text'])
    
    # Customize
    ax.set_yticks(range(len(methods_short)))
    ax.set_yticklabels(methods_short)
    ax.set_xlabel('Target Achievement Rate', fontweight='bold')
    ax.set_title('Fusion Experiment Success Summary', 
                fontsize=16, fontweight='bold', pad=20,
                color=ANTHROPIC_COLORS['text'])
    ax.set_xlim(0, 1.1)
    
    # Style
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_facecolor('white')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='#faf9f5', edgecolor='none')
        print(f"Success summary saved to: {save_path}")
    
    return fig
def main():
    parser = argparse.ArgumentParser(description="Create clean, target-focused fusion visualizations")
    parser.add_argument("--output_dir", default="analyze/fusion_analysis", help="Output directory")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", help="Output format")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df, targets = create_fusion_data()
    
    print("Creating clean, target-focused fusion visualizations...")
    
    # Create target-focused visualization showing specific values
    target_path = os.path.join(args.output_dir, f"fusion_targets_clean.{args.format}")
    fig1 = create_target_focused_visualization(df, targets, target_path)
    
    # Create compact results table
    table_path = os.path.join(args.output_dir, f"fusion_results_table.{args.format}")
    fig2 = create_compact_results_table(df, targets, table_path)
    
    # Create success summary
    summary_path = os.path.join(args.output_dir, f"fusion_success_clean.{args.format}")
    fig3 = create_success_summary(df, targets, summary_path)
    
    print("Clean visualizations completed!")
    print(f"Files saved in: {args.output_dir}")
    
    # Print summary of specific target values
    print("\n" + "="*60)
    print("TARGET DIMENSION RESULTS (Specific Values)")
    print("="*60)
    
    baseline = df.iloc[0, 1:].values
    dimensions = df.columns[1:].values
    
    for method in targets:
        if method in df['Method'].values:
            method_row = df[df['Method'] == method].iloc[0]
            method_values = method_row.iloc[1:].values
            effects = method_values - baseline
            
            method_name = method.replace('Fusion: ', '').replace('Subtraction: ', '')
            print(f"\n{method_name}:")
            
            for dim, direction, description in targets[method]:
                dim_idx = list(dimensions).index(dim)
                actual_effect = effects[dim_idx]
                baseline_val = baseline[dim_idx]
                new_val = method_values[dim_idx]
                
                arrow = '↗' if direction == 'increase' else '↘'
                success = (direction == 'increase' and actual_effect > 0.1) or (direction == 'decrease' and actual_effect < -0.1)
                status = '?' if success else '○'
                
                print(f"  {status} {dim} {arrow}: {baseline_val:.2f} → {new_val:.2f} (Δ{actual_effect:+.2f})")


if __name__ == "__main__":
    main()
