#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a summary table of BFI fusion evaluation results compared with baseline.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import re

def load_bfi_dimensions(file_path):
    """Load BFI dimension scores from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract mean scores for each dimension
        scores = {}
        for dimension, stats in data.items():
            if stats['mean'] is not None:
                scores[dimension] = stats['mean']
            else:
                scores[dimension] = np.nan
        
        return scores
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def parse_fusion_filename(filename):
    """Parse fusion filename to extract experiment type and coefficients."""
    # Pattern for baseline: baseline_bfi_dimensions.json
    if filename.startswith("baseline_"):
        return "baseline", None, None, None, None
    
    # Pattern for fusion: fusion_{trait1}_{coef1}_{trait2}_{coef2}_bfi_dimensions.json
    fusion_pattern = r'fusion_([^_]+)_([0-9.-]+)_([^_]+)_([0-9.-]+)_bfi_dimensions\.json'
    fusion_match = re.match(fusion_pattern, filename)
    
    if fusion_match:
        trait1 = fusion_match.group(1)
        coef1 = float(fusion_match.group(2))
        trait2 = fusion_match.group(3)
        coef2 = float(fusion_match.group(4))
        return "fusion", trait1, coef1, trait2, coef2
    
    # Pattern for subtraction: subtraction_{trait1}_{coef1}_{trait2}_{coef2}_bfi_dimensions.json
    subtraction_pattern = r'subtraction_([^_]+)_([0-9.-]+)_([^_]+)_([0-9.-]+)_bfi_dimensions\.json'
    subtraction_match = re.match(subtraction_pattern, filename)
    
    if subtraction_match:
        trait1 = subtraction_match.group(1)
        coef1 = float(subtraction_match.group(2))
        trait2 = subtraction_match.group(3)
        coef2 = float(subtraction_match.group(4))
        return "subtraction", trait1, coef1, trait2, coef2
    
    return None, None, None, None, None

def collect_baseline_results(baseline_dir):
    """Collect baseline results from baseline directory."""
    baseline_dir = Path(baseline_dir)
    baseline_scores = None
    
    # Look for baseline dimension file
    for json_file in baseline_dir.glob("**/baseline_*_dimensions.json"):
        scores = load_bfi_dimensions(json_file)
        if scores is not None:
            baseline_scores = scores
            break
    
    return baseline_scores

def collect_fusion_results(fusion_dir):
    """Collect all fusion results from fusion directory."""
    fusion_dir = Path(fusion_dir)
    results = {
        "fusion": {},
        "subtraction": {}
    }
    
    # Find all dimension JSON files
    for json_file in fusion_dir.glob("**/*_dimensions.json"):
        exp_type, trait1, coef1, trait2, coef2 = parse_fusion_filename(json_file.name)
        
        if exp_type is None or exp_type == "baseline":
            continue
        
        scores = load_bfi_dimensions(json_file)
        if scores is None:
            continue
        
        # Create experiment key
        if exp_type == "fusion":
            exp_key = f"{trait1}+{trait2}"
            coef_key = f"{coef1}+{coef2}"
        else:  # subtraction
            exp_key = f"{trait1}-{trait2}"
            coef_key = f"{coef1}{coef2}"  # coef2 should already be negative
        
        if exp_key not in results[exp_type]:
            results[exp_type][exp_key] = {}
        
        results[exp_type][exp_key][coef_key] = scores
    
    return results

def create_fusion_summary_table(baseline_scores, fusion_results):
    """Create a summary table of fusion results compared with baseline."""
    
    # Define the Big Five dimensions
    dimensions = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    
    # Define expected experiment categories
    same_dimension_experiments = [
        "dependable+careless",      # conscientiousness neutralization
        "outgoing+solitary",        # extraversion neutralization
        "nervous+calm"              # neuroticism neutralization
    ]
    
    cross_dimension_experiments = [
        "inventive+outgoing",       # openness + extraversion
        "dependable+compassionate", # conscientiousness + agreeableness
        "calm+outgoing"             # neuroticism + extraversion
    ]
    
    enhanced_subtraction_experiments = [
        "inventive-consistent",     # enhance openness
        "outgoing-solitary",        # enhance extraversion
        "dependable-selfinterested" # enhance agreeableness (note: self-interested becomes selfinterested in filename)
    ]
    
    cross_subtraction_experiments = [
        "inventive-outgoing",       # maintain openness, suppress extraversion
        "dependable-compassionate"  # maintain conscientiousness, suppress agreeableness
    ]
    
    # Create column structure
    columns = ["Baseline"]
    
    # Add fusion experiments
    for exp in same_dimension_experiments + cross_dimension_experiments:
        columns.append(f"Fusion: {exp}")
    
    # Add subtraction experiments
    for exp in enhanced_subtraction_experiments + cross_subtraction_experiments:
        columns.append(f"Subtraction: {exp}")
    
    # Initialize DataFrame
    summary_df = pd.DataFrame(index=dimensions, columns=columns)
    
    # Fill baseline scores
    if baseline_scores:
        for dim in dimensions:
            if dim in baseline_scores:
                summary_df.loc[dim, "Baseline"] = baseline_scores[dim]
    
    # Fill fusion experiment scores
    if "fusion" in fusion_results:
        for exp_key, coef_data in fusion_results["fusion"].items():
            # Use the first coefficient combination found (or we could aggregate)
            if coef_data:
                first_coef = list(coef_data.keys())[0]
                scores = coef_data[first_coef]
                col_name = f"Fusion: {exp_key}"
                if col_name in columns:
                    for dim in dimensions:
                        if dim in scores:
                            summary_df.loc[dim, col_name] = scores[dim]
    
    # Fill subtraction experiment scores
    if "subtraction" in fusion_results:
        for exp_key, coef_data in fusion_results["subtraction"].items():
            # Use the first coefficient combination found
            if coef_data:
                first_coef = list(coef_data.keys())[0]
                scores = coef_data[first_coef]
                col_name = f"Subtraction: {exp_key}"
                if col_name in columns:
                    for dim in dimensions:
                        if dim in scores:
                            summary_df.loc[dim, col_name] = scores[dim]
    
    return summary_df

def create_delta_table(baseline_scores, fusion_results):
    """Create a table showing differences from baseline."""
    
    summary_df = create_fusion_summary_table(baseline_scores, fusion_results)
    
    if "Baseline" not in summary_df.columns or baseline_scores is None:
        print("Warning: No baseline results found for delta calculation")
        return summary_df
    
    # Calculate deltas
    delta_df = summary_df.copy()
    baseline_col = summary_df["Baseline"]
    
    for col in summary_df.columns:
        if col != "Baseline":
            delta_df[col] = summary_df[col] - baseline_col
    
    # Remove baseline column from delta table
    delta_df = delta_df.drop("Baseline", axis=1)
    
    return delta_df

def create_detailed_fusion_table(baseline_scores, fusion_results):
    """Create a detailed table with all coefficient combinations."""
    
    # Define the Big Five dimensions
    dimensions = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    
    # Collect all experiments and their coefficient combinations
    column_tuples = [("Baseline", "")]
    
    # Add fusion experiments
    if "fusion" in fusion_results:
        for exp_key, coef_data in sorted(fusion_results["fusion"].items()):
            for coef_key in sorted(coef_data.keys()):
                column_tuples.append((f"Fusion: {exp_key}", coef_key))
    
    # Add subtraction experiments
    if "subtraction" in fusion_results:
        for exp_key, coef_data in sorted(fusion_results["subtraction"].items()):
            for coef_key in sorted(coef_data.keys()):
                column_tuples.append((f"Subtraction: {exp_key}", coef_key))
    
    multi_columns = pd.MultiIndex.from_tuples(column_tuples, names=['Experiment', 'Coefficients'])
    
    # Initialize DataFrame
    detailed_df = pd.DataFrame(index=dimensions, columns=multi_columns)
    
    # Fill baseline scores
    if baseline_scores:
        for dim in dimensions:
            if dim in baseline_scores:
                detailed_df.loc[dim, ("Baseline", "")] = baseline_scores[dim]
    
    # Fill fusion experiment scores
    if "fusion" in fusion_results:
        for exp_key, coef_data in fusion_results["fusion"].items():
            for coef_key, scores in coef_data.items():
                for dim in dimensions:
                    if dim in scores:
                        detailed_df.loc[dim, (f"Fusion: {exp_key}", coef_key)] = scores[dim]
    
    # Fill subtraction experiment scores
    if "subtraction" in fusion_results:
        for exp_key, coef_data in fusion_results["subtraction"].items():
            for coef_key, scores in coef_data.items():
                for dim in dimensions:
                    if dim in scores:
                        detailed_df.loc[dim, (f"Subtraction: {exp_key}", coef_key)] = scores[dim]
    
    return detailed_df

def format_table_for_latex(df, precision=2):
    """Format table for LaTeX output."""
    formatted_df = df.round(precision)
    
    # Replace NaN with dash
    formatted_df = formatted_df.fillna("--")
    
    return formatted_df

def save_table(df, output_path, format="csv"):
    """Save table in specified format."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if format.lower() == "csv":
        df.astype(float).round(2).to_csv(output_path)
    elif format.lower() == "latex":
        latex_str = df.to_latex(float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "--")
        with open(output_path, 'w') as f:
            f.write(latex_str)
    elif format.lower() == "excel":
        df.to_excel(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def print_fusion_summary_stats(baseline_scores, fusion_results):
    """Print summary statistics for fusion experiments."""
    print("\n" + "="*60)
    print("BFI FUSION EVALUATION SUMMARY")
    print("="*60)
    
    # Check baseline
    if baseline_scores:
        print("? Baseline results found")
        print(f"Baseline dimensions: {', '.join(baseline_scores.keys())}")
    else:
        print("? No baseline results found")
    
    # Count fusion experiments
    fusion_count = len(fusion_results.get("fusion", {}))
    subtraction_count = len(fusion_results.get("subtraction", {}))
    
    print(f"Fusion experiments: {fusion_count}")
    if fusion_count > 0:
        fusion_experiments = list(fusion_results["fusion"].keys())
        print(f"  - {', '.join(fusion_experiments)}")
    
    print(f"Subtraction experiments: {subtraction_count}")
    if subtraction_count > 0:
        subtraction_experiments = list(fusion_results["subtraction"].keys())
        print(f"  - {', '.join(subtraction_experiments)}")
    
    total_combinations = 0
    for exp_type in ["fusion", "subtraction"]:
        for exp_key, coef_data in fusion_results.get(exp_type, {}).items():
            total_combinations += len(coef_data)
    
    print(f"Total coefficient combinations: {total_combinations}")

def main():
    parser = argparse.ArgumentParser(description="Generate BFI fusion evaluation summary table")
    parser.add_argument("--baseline_dir", default="eval_bfi_output/Qwen2.5-7B-Instruct/", 
                       help="Directory containing baseline BFI results")
    parser.add_argument("--fusion_dir", default="eval_fusion_bfi_output/Qwen2.5-7B-Instruct/", 
                       help="Directory containing fusion BFI results")
    parser.add_argument("--output_dir", default="analyze/fusion_analysis", 
                       help="Output directory for analysis results")
    parser.add_argument("--format", choices=["csv", "latex", "excel"], default="csv", 
                       help="Output format")
    parser.add_argument("--detailed", action="store_true", 
                       help="Create detailed table with all coefficient combinations")
    parser.add_argument("--delta", action="store_true", 
                       help="Create delta table (differences from baseline)")
    
    args = parser.parse_args()
    
    # Check directories exist
    if not os.path.exists(args.baseline_dir):
        print(f"Error: Baseline directory not found: {args.baseline_dir}")
        return
    
    if not os.path.exists(args.fusion_dir):
        print(f"Error: Fusion directory not found: {args.fusion_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect results
    print(f"Collecting baseline results from: {args.baseline_dir}")
    baseline_scores = collect_baseline_results(args.baseline_dir)
    
    print(f"Collecting fusion results from: {args.fusion_dir}")
    fusion_results = collect_fusion_results(args.fusion_dir)
    
    if not fusion_results or (not fusion_results.get("fusion") and not fusion_results.get("subtraction")):
        print("No fusion/subtraction results found!")
        return
    
    # Print summary
    print_fusion_summary_stats(baseline_scores, fusion_results)
    
    # Create tables
    if args.detailed:
        print("\nCreating detailed multi-level table...")
        table = create_detailed_fusion_table(baseline_scores, fusion_results)
        output_path = os.path.join(args.output_dir, f"fusion_detailed.{args.format}")
    elif args.delta:
        print("\nCreating delta table...")
        table = create_delta_table(baseline_scores, fusion_results)
        output_path = os.path.join(args.output_dir, f"fusion_delta.{args.format}")
    else:
        print("\nCreating summary table...")
        table = create_fusion_summary_table(baseline_scores, fusion_results)
        output_path = os.path.join(args.output_dir, f"fusion_summary.{args.format}")
    
    # Save table
    save_table(table, output_path, args.format)
    print(f"Table saved to: {output_path}")
    
    # Display preview
    print(f"\nTable preview:")
    print(table.round(2))

if __name__ == "__main__":
    main()
