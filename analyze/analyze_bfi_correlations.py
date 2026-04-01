#!/usr/bin/env python3
# -*- coding: gb2312 -*-
"""
Analyze BFI summary table to demonstrate linear correlation between 
persona vector steering strength and corresponding BFI dimensions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define the mapping between persona vectors and BFI dimensions
PERSONA_TO_BFI = {
    'inventive': 'Openness',
    'consistent': 'Openness', 
    'dependable': 'Conscientiousness',
    'careless': 'Conscientiousness',
    'outgoing': 'Extraversion',
    'solitary': 'Extraversion',
    'compassionate': 'Agreeableness', 
    'aloof': 'Agreeableness',
    'nervous': 'Neuroticism',
    'calm': 'Neuroticism'
}

# Define expected direction of correlation (positive or negative)
EXPECTED_DIRECTION = {
    'inventive': 1,    # Higher inventive -> Higher Openness
    'consistent': -1,  # Higher consistent -> Lower Openness
    'dependable': 1,  # Higher dependable -> Higher Conscientiousness
    'careless': -1,    # Higher careless -> Lower Conscientiousness
    'outgoing': 1,     # Higher outgoing -> Higher Extraversion
    'solitary': -1,    # Higher solitary -> Lower Extraversion
    'compassionate': 1,  # Higher compassionate -> Higher Agreeableness
    'aloof': -1, # Higher aloof -> Lower Agreeableness
    'nervous': 1,      # Higher nervous -> Higher Neuroticism
    'calm': -1         # Higher calm -> Lower Neuroticism
}

class BFIAnalyzer:
    def __init__(self, csv_path: str):
        """Initialize analyzer with BFI summary table."""
        self.df = pd.read_csv(csv_path, index_col=0)
        self.baseline_scores = self.df['Baseline'].to_dict()
        self.results = {}
        
    def extract_steering_data(self):
        """Extract steering data for each persona vector."""
        steering_data = {}
        
        for col in self.df.columns:
            if col == 'Baseline':
                continue
                
            # Parse column name (e.g., 'inventive_0.25')
            parts = col.split('_')
            if len(parts) == 2:
                persona = parts[0]
                strength = float(parts[1])
                
                if persona not in steering_data:
                    steering_data[persona] = {'strengths': [], 'scores': {}}
                
                steering_data[persona]['strengths'].append(strength)
                
                # Get scores for all BFI dimensions
                for dimension in self.df.index:
                    if dimension not in steering_data[persona]['scores']:
                        steering_data[persona]['scores'][dimension] = []
                    steering_data[persona]['scores'][dimension].append(self.df.loc[dimension, col])
        
        # Sort by strength for consistent ordering
        for persona in steering_data:
            sorted_indices = np.argsort(steering_data[persona]['strengths'])
            steering_data[persona]['strengths'] = [steering_data[persona]['strengths'][i] for i in sorted_indices]
            for dimension in steering_data[persona]['scores']:
                steering_data[persona]['scores'][dimension] = [steering_data[persona]['scores'][dimension][i] for i in sorted_indices]
        
        return steering_data
    
    def calculate_correlations(self, steering_data):
        """Calculate correlations between steering strength and BFI scores."""
        correlations = {}
        
        for persona, data in steering_data.items():
            correlations[persona] = {}
            target_dimension = PERSONA_TO_BFI[persona]
            expected_direction = EXPECTED_DIRECTION[persona]
            
            for dimension in self.df.index:
                strengths = data['strengths']
                scores = data['scores'][dimension]
                
                # Calculate Pearson correlation
                r, p_value = stats.pearsonr(strengths, scores)
                
                # Calculate R-squared
                r_squared = r ** 2
                
                # Calculate slope using linear regression
                slope, intercept = np.polyfit(strengths, scores, 1)
                
                # Calculate directional correlation (considering expected direction)
                directional_r = r * expected_direction if dimension == target_dimension else r
                
                correlations[persona][dimension] = {
                    'r': r,
                    'r_squared': r_squared,
                    'p_value': p_value,
                    'slope': slope,
                    'intercept': intercept,
                    'directional_r': directional_r,
                    'is_target': dimension == target_dimension,
                    'significant': p_value < 0.05
                }
        
        return correlations
    
    def create_correlation_matrix(self, correlations, output_dir: Path):
        """Create correlation matrix heatmap with product of persona index and correlation strength (slope)."""
        # Create matrix for target dimension correlations
        personas = list(correlations.keys())
        dimensions = list(self.df.index)
        
        # Matrix of product: persona_index * slope
        corr_matrix = np.zeros((len(personas), len(dimensions)))
        p_values = np.zeros((len(personas), len(dimensions)))
        slopes = np.zeros((len(personas), len(dimensions)))
        
        for i, persona in enumerate(personas):
            for j, dimension in enumerate(dimensions):
                slope = correlations[persona][dimension]['slope']
                # Product of persona index (i+1 to avoid zero) and slope
                corr_matrix[i, j] = (i + 1) * slope
                p_values[i, j] = correlations[persona][dimension]['p_value']
                slopes[i, j] = slope
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create mask for non-significant correlations
        mask = p_values >= 0.05
        
        # Use a symmetric colormap centered at 0
        vmax = np.abs(corr_matrix).max()
        vmin = -vmax
        
        sns.heatmap(corr_matrix, 
                   xticklabels=dimensions,
                   yticklabels=personas,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r',
                   center=0,
                   vmin=vmin, vmax=vmax,
                   mask=mask,
                   cbar_kws={'label': 'Persona Index × Correlation Strength (Slope)'},
                   ax=ax)
        
        plt.title('Product of Persona Index and Correlation Strength (Slope)\n(Only significant correlations shown, p < 0.05)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('BFI Dimensions', fontweight='bold')
        plt.ylabel('Persona Vectors', fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'correlation_matrix.pdf', bbox_inches='tight')
        plt.close()
    
    def create_target_correlation_plot(self, steering_data, correlations, output_dir: Path):
        """Create scatter plots showing correlation for target dimensions."""
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        personas = list(steering_data.keys())
        
        for i, persona in enumerate(personas):
            ax = axes[i]
            target_dimension = PERSONA_TO_BFI[persona]
            
            strengths = steering_data[persona]['strengths']
            target_scores = steering_data[persona]['scores'][target_dimension]
            baseline_score = self.baseline_scores[target_dimension]
            
            # Create scatter plot
            ax.scatter(strengths, target_scores, alpha=0.7, s=60)
            
            # Add regression line
            z = np.polyfit(strengths, target_scores, 1)
            p = np.poly1d(z)
            ax.plot(strengths, p(strengths), "r--", alpha=0.8, linewidth=2)
            
            # Add baseline line
            ax.axhline(y=baseline_score, color='gray', linestyle=':', alpha=0.7, label='Baseline')
            
            # Get correlation info
            corr_info = correlations[persona][target_dimension]
            r = corr_info['r']
            r_squared = corr_info['r_squared']
            p_value = corr_info['p_value']
            
            # Set title and labels
            ax.set_title(f'{persona.title()} → {target_dimension}\nr = {r:.3f}, R? = {r_squared:.3f}\np = {p_value:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Steering Strength')
            ax.set_ylabel(f'{target_dimension} Score')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Linear Correlation: Persona Vector Strength vs Target BFI Dimension', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(output_dir / 'target_correlations.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'target_correlations.pdf', bbox_inches='tight')
        plt.close()
    
    def create_selectivity_analysis(self, correlations, output_dir: Path):
        """Analyze and visualize selectivity - how specifically each vector affects its target dimension."""
        selectivity_data = []
        
        for persona in correlations:
            target_dimension = PERSONA_TO_BFI[persona]
            target_r = correlations[persona][target_dimension]['r']
            target_r_squared = correlations[persona][target_dimension]['r_squared']
            
            # Calculate average correlation with non-target dimensions
            non_target_correlations = []
            for dimension in correlations[persona]:
                if dimension != target_dimension:
                    non_target_correlations.append(abs(correlations[persona][dimension]['r']))
            
            avg_non_target_r = np.mean(non_target_correlations)
            max_non_target_r = np.max(non_target_correlations)
            
            # Selectivity ratio: target correlation vs average non-target
            selectivity_ratio = abs(target_r) / (avg_non_target_r + 1e-6)  # Add small value to avoid division by zero
            
            selectivity_data.append({
                'persona': persona,
                'target_dimension': target_dimension,
                'target_r': target_r,
                'target_r_squared': target_r_squared,
                'avg_non_target_r': avg_non_target_r,
                'max_non_target_r': max_non_target_r,
                'selectivity_ratio': selectivity_ratio
            })
        
        selectivity_df = pd.DataFrame(selectivity_data)
        
        # Create selectivity visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Target vs Non-target correlations
        ax1.scatter(selectivity_df['avg_non_target_r'], selectivity_df['target_r'].abs(), 
                   s=100, alpha=0.7)
        
        for i, row in selectivity_df.iterrows():
            ax1.annotate(row['persona'], 
                        (row['avg_non_target_r'], abs(row['target_r'])),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlabel('Average |Correlation| with Non-Target Dimensions')
        ax1.set_ylabel('|Correlation| with Target Dimension')
        ax1.set_title('Selectivity: Target vs Non-Target Correlations')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line (perfect selectivity would be high target, low non-target)
        max_val = max(ax1.get_xlim()[1], ax1.get_ylim()[1])
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal correlation')
        ax1.legend()
        
        # Plot 2: Selectivity ratios
        colors = sns.color_palette("husl", len(selectivity_df))
        bars = ax2.bar(selectivity_df['persona'], selectivity_df['selectivity_ratio'], color=colors)
        ax2.set_xlabel('Persona Vector')
        ax2.set_ylabel('Selectivity Ratio')
        ax2.set_title('Selectivity Ratio (Target/Average Non-Target)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at ratio = 1
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal selectivity')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'selectivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'selectivity_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        return selectivity_df
    
    def generate_correlation_summary_table(self, correlations, selectivity_df, output_dir: Path):
        """Generate a summary table of all correlations."""
        summary_data = []
        
        for persona in correlations:
            target_dimension = PERSONA_TO_BFI[persona]
            target_info = correlations[persona][target_dimension]
            
            # Find selectivity info
            selectivity_info = selectivity_df[selectivity_df['persona'] == persona].iloc[0]
            
            summary_data.append({
                'Persona Vector': persona,
                'Target Dimension': target_dimension,
                'Target Correlation (r)': f"{target_info['r']:.3f}",
                'Target R?': f"{target_info['r_squared']:.3f}",
                'Target p-value': f"{target_info['p_value']:.3f}",
                'Significant': 'Yes' if target_info['significant'] else 'No',
                'Avg Non-Target |r|': f"{selectivity_info['avg_non_target_r']:.3f}",
                'Selectivity Ratio': f"{selectivity_info['selectivity_ratio']:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'correlation_summary.csv', index=False)
        
        return summary_df
    
    def run_full_analysis(self, output_dir: str = 'bfi_analysis'):
        """Run complete analysis and generate all outputs."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("? Extracting steering data...")
        steering_data = self.extract_steering_data()
        
        print("? Calculating correlations...")
        correlations = self.calculate_correlations(steering_data)
        
        print("? Creating correlation matrix heatmap...")
        self.create_correlation_matrix(correlations, output_path)
        
        print("? Creating target correlation plots...")
        self.create_target_correlation_plot(steering_data, correlations, output_path)
        
        print("? Analyzing selectivity...")
        selectivity_df = self.create_selectivity_analysis(correlations, output_path)
        
        print("? Generating summary table...")
        summary_df = self.generate_correlation_summary_table(correlations, selectivity_df, output_path)
        
        # Print summary results
        print("\n" + "="*60)
        print("? SUMMARY: Persona Vector Selectivity Analysis")
        print("="*60)
        
        print(f"\n? Target Dimension Correlations:")
        for _, row in summary_df.iterrows():
            persona = row['Persona Vector']
            target = row['Target Dimension']
            r = row['Target Correlation (r)']
            significant = row['Significant']
            selectivity = row['Selectivity Ratio']
            
            status = "?" if significant == "Yes" else "?"
            print(f"{status} {persona:12} → {target:15} | r = {r:7} | Selectivity = {selectivity:5}")
        
        print(f"\n? Strong Correlations (|r| > 0.7):")
        strong_correlations = []
        for persona in correlations:
            target_dim = PERSONA_TO_BFI[persona]
            r = correlations[persona][target_dim]['r']
            if abs(r) > 0.7:
                strong_correlations.append((persona, target_dim, r))
        
        if strong_correlations:
            for persona, dim, r in strong_correlations:
                print(f"  ? {persona} → {dim}: r = {r:.3f}")
        else:
            print("  No correlations with |r| > 0.7 found")
        
        print(f"\n? High Selectivity (ratio > 2.0):")
        high_selectivity = selectivity_df[selectivity_df['selectivity_ratio'] > 2.0]
        if not high_selectivity.empty:
            for _, row in high_selectivity.iterrows():
                print(f"  ? {row['persona']}: {row['selectivity_ratio']:.2f}x more selective")
        else:
            print("  No persona vectors with selectivity ratio > 2.0")
        
        print(f"\n? All outputs saved to: {output_path.absolute()}")
        print("   ? correlation_matrix.png/pdf - Heatmap of all correlations")
        print("   ? target_correlations.png/pdf - Individual target dimension plots")
        print("   ? selectivity_analysis.png/pdf - Selectivity analysis")
        print("   ? correlation_summary.csv - Summary statistics table")
        
        return steering_data, correlations, selectivity_df, summary_df

def main():
    parser = argparse.ArgumentParser(description='Analyze BFI persona vector correlations')
    parser.add_argument('--input', default='analyze/bfi_summary_table.csv',
                       help='Path to BFI summary table CSV')
    parser.add_argument('--output', default='analyze/bfi_correlation_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"? Error: Input file {args.input} not found")
        return
    
    print(f"? Starting BFI Persona Vector Analysis")
    print(f"? Input: {args.input}")
    print(f"? Output: {args.output}")
    
    analyzer = BFIAnalyzer(args.input)
    steering_data, correlations, selectivity_df, summary_df = analyzer.run_full_analysis(args.output)
    
    print("\n? Analysis complete!")

if __name__ == "__main__":
    main()
