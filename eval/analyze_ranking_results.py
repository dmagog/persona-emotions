#!/usr/bin/env python3
"""
Analyze and visualize ranking evaluation results.

This script provides comprehensive analysis of the ranking evaluation results,
including statistical tests, visualizations, and detailed breakdowns.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import argparse
from typing import Dict, List, Any
import pandas as pd

class RankingAnalyzer:
    """Analyzer for ranking evaluation results"""
    
    def __init__(self, results_file: str):
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        self.sessions = self.data['sessions']
        self.summary = self.data['summary']
    
    def analyze_significance(self) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        
        metrics = ['trait_adherence', 'role_consistency', 'style_conformance', 
                  'engagement', 'insightfulness', 'overall']
        
        results = {}
        
        for metric in metrics:
            # Get win rates for each session
            win_rates = [session['win_rates'][metric] for session in self.sessions]
            
            # One-sample t-test against 0.5 (null hypothesis: no difference)
            t_stat, p_value = stats.ttest_1samp(win_rates, 0.5)
            
            # Effect size (Cohen's d)
            mean_rate = np.mean(win_rates)
            std_rate = np.std(win_rates, ddof=1)
            cohens_d = (mean_rate - 0.5) / std_rate if std_rate > 0 else 0
            
            # Confidence interval for the mean
            n = len(win_rates)
            se = std_rate / np.sqrt(n)
            ci_95 = stats.t.interval(0.95, n-1, mean_rate, se)
            
            results[metric] = {
                'mean_win_rate': mean_rate,
                'std_win_rate': std_rate,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'confidence_interval_95': ci_95,
                'significant': p_value < 0.05,
                'favors_steered': mean_rate > 0.5
            }
        
        return results
    
    def create_visualizations(self, output_dir: str):
        """Create comprehensive visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overall win rates bar chart
        self._plot_overall_win_rates(output_path)
        
        # 2. Session-level win rate distributions
        self._plot_win_rate_distributions(output_path)
        
        # 3. Win rate heatmap by persona
        self._plot_persona_heatmap(output_path)
        
        # 4. Turn-by-turn analysis
        self._plot_turn_analysis(output_path)
        
        # 5. Statistical significance visualization
        self._plot_significance(output_path)
        
        print(f"Visualizations saved to: {output_path}")
    
    def _plot_overall_win_rates(self, output_path: Path):
        """Plot overall win rates"""
        
        metrics = ['trait_adherence', 'role_consistency', 'style_conformance', 
                  'engagement', 'insightfulness', 'overall']
        rates = [self.summary['overall_win_rates'][metric] for metric in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#1f77b4' if rate > 0.5 else '#ff7f0e' for rate in rates]
        bars = ax.bar(range(len(metrics)), rates, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Add reference line at 0.5
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='No difference (50%)')
        
        ax.set_xlabel('Evaluation Criteria', fontsize=12)
        ax.set_ylabel('Steered Win Rate', fontsize=12)
        ax.set_title('Overall Win Rates: Steered vs Non-steered Responses', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'overall_win_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_win_rate_distributions(self, output_path: Path):
        """Plot distributions of session-level win rates"""
        
        metrics = ['trait_adherence', 'role_consistency', 'style_conformance', 
                  'engagement', 'insightfulness', 'overall']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            win_rates = [session['win_rates'][metric] for session in self.sessions]
            
            ax = axes[i]
            ax.hist(win_rates, bins=20, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add normal distribution overlay
            mu, sigma = np.mean(win_rates), np.std(win_rates)
            x = np.linspace(0, 1, 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal (¦Ì={mu:.2f}, ¦Ò={sigma:.2f})')
            
            # Add reference line at 0.5
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='No difference')
            ax.axvline(x=mu, color='green', linestyle='-', alpha=0.7, label=f'Mean ({mu:.1%})')
            
            ax.set_xlabel('Win Rate', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Distribution of Session-Level Win Rates', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'win_rate_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_persona_heatmap(self, output_path: Path):
        """Plot win rates by persona"""
        
        # Create persona-metric matrix
        personas = list(set(session['persona_name'] for session in self.sessions))
        metrics = ['trait_adherence', 'role_consistency', 'style_conformance', 
                  'engagement', 'insightfulness', 'overall']
        
        # Calculate win rates by persona
        persona_data = {}
        for persona in personas:
            persona_sessions = [s for s in self.sessions if s['persona_name'] == persona]
            persona_data[persona] = {}
            
            for metric in metrics:
                total_wins = sum(s['steered_wins'][metric] for s in persona_sessions)
                total_turns = sum(s['num_turns'] for s in persona_sessions)
                win_rate = total_wins / total_turns if total_turns > 0 else 0
                persona_data[persona][metric] = win_rate
        
        # Create DataFrame for heatmap
        df = pd.DataFrame(persona_data).T
        df = df[metrics]  # Ensure column order
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0.5,
                   cbar_kws={'label': 'Steered Win Rate'}, ax=ax)
        
        ax.set_xlabel('Evaluation Criteria', fontsize=12)
        ax.set_ylabel('Persona', fontsize=12)
        ax.set_title('Win Rates by Persona and Criteria', fontsize=14, fontweight='bold')
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'persona_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_turn_analysis(self, output_path: Path):
        """Analyze performance by turn number"""
        
        # Collect turn-level data
        turn_data = []
        for session in self.sessions:
            for turn in session['turns']:
                turn_data.append({
                    'turn_number': turn['turn_number'],
                    'trait_adherence': turn['steered_wins']['trait_adherence'],
                    'role_consistency': turn['steered_wins']['role_consistency'],
                    'style_conformance': turn['steered_wins']['style_conformance'],
                    'engagement': turn['steered_wins']['engagement'],
                    'insightfulness': turn['steered_wins']['insightfulness'],
                    'overall': turn['steered_wins']['overall']
                })
        
        df = pd.DataFrame(turn_data)
        
        # Calculate win rates by turn number
        turn_win_rates = df.groupby('turn_number').mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['trait_adherence', 'role_consistency', 'style_conformance', 
                  'engagement', 'insightfulness', 'overall']
        
        for metric in metrics:
            if metric in turn_win_rates.columns:
                ax.plot(turn_win_rates.index, turn_win_rates[metric], 
                       marker='o', label=metric.replace('_', ' ').title(), linewidth=2)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='No difference')
        ax.set_xlabel('Turn Number', fontsize=12)
        ax.set_ylabel('Steered Win Rate', fontsize=12)
        ax.set_title('Win Rates by Turn Number', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path / 'turn_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_significance(self, output_path: Path):
        """Plot statistical significance results"""
        
        sig_results = self.analyze_significance()
        
        metrics = list(sig_results.keys())
        means = [sig_results[m]['mean_win_rate'] for m in metrics]
        cis = [sig_results[m]['confidence_interval_95'] for m in metrics]
        p_values = [sig_results[m]['p_value'] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create error bars
        lower_errors = [mean - ci[0] for mean, ci in zip(means, cis)]
        upper_errors = [ci[1] - mean for mean, ci in zip(means, cis)]
        
        colors = ['green' if p < 0.05 else 'orange' for p in p_values]
        
        bars = ax.bar(range(len(metrics)), means, yerr=[lower_errors, upper_errors],
                     capsize=5, color=colors, alpha=0.7, ecolor='black')
        
        # Add significance indicators
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            height = bar.get_height()
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   significance, ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='No difference (H?)')
        
        ax.set_xlabel('Evaluation Criteria', fontsize=12)
        ax.set_ylabel('Mean Win Rate ¡À 95% CI', fontsize=12)
        ax.set_title('Statistical Significance of Steered vs Non-steered Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add legend for significance levels
        ax.text(0.02, 0.98, '*** p < 0.001\n** p < 0.01\n* p < 0.05\nns: not significant', 
               transform=ax.transAxes, verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'significance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str):
        """Generate a comprehensive text report"""
        
        sig_results = self.analyze_significance()
        
        with open(output_file, 'w') as f:
            f.write("DIALOG RANKING EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Sessions Evaluated: {self.summary['total_sessions']}\n")
            f.write(f"Total Turns Compared: {self.summary['total_turns']}\n\n")
            
            f.write("OVERALL RESULTS\n")
            f.write("-" * 20 + "\n")
            
            for metric, rate in self.summary['overall_win_rates'].items():
                sig_info = sig_results[metric]
                significance = "***" if sig_info['p_value'] < 0.001 else "**" if sig_info['p_value'] < 0.01 else "*" if sig_info['p_value'] < 0.05 else ""
                
                f.write(f"{metric.replace('_', ' ').title()}: {rate:.1%} {significance}\n")
                f.write(f"  - Statistical significance: p = {sig_info['p_value']:.4f}\n")
                f.write(f"  - Effect size (Cohen's d): {sig_info['cohens_d']:.3f}\n")
                f.write(f"  - 95% CI: [{sig_info['confidence_interval_95'][0]:.1%}, {sig_info['confidence_interval_95'][1]:.1%}]\n")
                f.write(f"  - Favors: {'Steered' if sig_info['favors_steered'] else 'Non-steered'}\n\n")
            
            f.write("INTERPRETATION\n")
            f.write("-" * 20 + "\n")
            
            # Count significant improvements
            significant_improvements = sum(1 for m in sig_results.values() 
                                         if m['significant'] and m['favors_steered'])
            significant_degradations = sum(1 for m in sig_results.values() 
                                         if m['significant'] and not m['favors_steered'])
            
            f.write(f"Metrics with significant improvement: {significant_improvements}/6\n")
            f.write(f"Metrics with significant degradation: {significant_degradations}/6\n\n")
            
            if significant_improvements > significant_degradations:
                f.write("CONCLUSION: Steering shows overall positive impact on response quality.\n")
            elif significant_degradations > significant_improvements:
                f.write("CONCLUSION: Steering shows overall negative impact on response quality.\n")
            else:
                f.write("CONCLUSION: Steering shows mixed or neutral impact on response quality.\n")
            
            f.write("\nBest performing metrics for steering:\n")
            sorted_metrics = sorted(sig_results.items(), 
                                  key=lambda x: x[1]['mean_win_rate'], reverse=True)
            for metric, result in sorted_metrics[:3]:
                f.write(f"  - {metric.replace('_', ' ').title()}: {result['mean_win_rate']:.1%}\n")
            
        print(f"Report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ranking evaluation results')
    parser.add_argument('--results', required=True, help='Path to ranking results JSON file')
    parser.add_argument('--output_dir', default='ranking_analysis', help='Directory to save analysis outputs')
    parser.add_argument('--report', help='Path to save text report (optional)')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = RankingAnalyzer(args.results)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    analyzer.create_visualizations(args.output_dir)
    
    # Generate report
    if args.report:
        print("Generating report...")
        analyzer.generate_report(args.report)
    else:
        report_path = Path(args.output_dir) / 'ranking_report.txt'
        analyzer.generate_report(str(report_path))
    
    # Print summary
    sig_results = analyzer.analyze_significance()
    print("\nQuick Summary:")
    print("-" * 30)
    for metric, result in sig_results.items():
        status = "? Significant" if result['significant'] else "¡ð Not significant"
        direction = "¡ü Steered better" if result['favors_steered'] else "¡ý Non-steered better"
        print(f"{metric.replace('_', ' ').title()}: {result['mean_win_rate']:.1%} | {status} | {direction}")

if __name__ == "__main__":
    main()
