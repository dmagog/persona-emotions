#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a summary table of BFI evaluation results.
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

def parse_filename(filename):
    """Parse filename to extract trait and coefficient."""
    # Pattern for baseline: baseline_bfi_dimensions.json
    if filename.startswith("baseline_"):
        return "baseline", 0.0
    
    # Pattern for persona vectors: {trait}_coef{coef}_bfi_dimensions.json
    pattern = r'([^_]+)_coef([0-9.]+)_bfi_dimensions\.json'
    match = re.match(pattern, filename)
    
    if match:
        trait = match.group(1)
        coef = float(match.group(2))
        return trait, coef
    
    return None, None

def collect_bfi_results(results_dir):
    """Collect all BFI results from a directory."""
    results_dir = Path(results_dir)
    results = {}
    
    # Find all dimension JSON files
    for json_file in results_dir.glob("*_dimensions.json"):
        trait, coef = parse_filename(json_file.name)
        
        if trait is None:
            continue
        
        scores = load_bfi_dimensions(json_file)
        if scores is None:
            continue
        
        if trait not in results:
            results[trait] = {}
        
        results[trait][coef] = scores
    
    return results

def create_bfi_summary_table(results, coefficients=None):
    """Create a summary table of BFI results."""
    
    # Define the Big Five dimensions
    dimensions = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    
    # Define trait pairs and order
    trait_pairs = [
        ("inventive", "consistent"),
        ("dependable", "careless"),
        ("outgoing", "solitary"),
        ("compassionate", "self-interested"),
        ("nervous", "calm")
    ]
    
    # Get all available coefficients if not specified
    if coefficients is None:
        all_coefs = set()
        for trait_data in results.values():
            all_coefs.update(trait_data.keys())
        coefficients = sorted([c for c in all_coefs if c != 0.0])  # Exclude baseline (0.0)
    
    # Create column structure - grouped by trait, then by coefficient
    columns = ["Baseline"]
    for trait1, trait2 in trait_pairs:
        # Add all coefficients for trait1 first
        for coef in coefficients:
            columns.append(f"{trait1}_{coef}")
        # Then add all coefficients for trait2
        for coef in coefficients:
            columns.append(f"{trait2}_{coef}")
    
    # Initialize DataFrame
    summary_df = pd.DataFrame(index=dimensions, columns=columns)
    
    # Fill baseline scores
    if "baseline" in results and 0.0 in results["baseline"]:
        baseline_scores = results["baseline"][0.0]
        for dim in dimensions:
            if dim in baseline_scores:
                summary_df.loc[dim, "Baseline"] = baseline_scores[dim]
    
    # Fill persona vector scores
    for trait1, trait2 in trait_pairs:
        # Fill trait1 scores
        if trait1 in results:
            for coef in coefficients:
                if coef in results[trait1]:
                    trait1_scores = results[trait1][coef]
                    for dim in dimensions:
                        if dim in trait1_scores:
                            summary_df.loc[dim, f"{trait1}_{coef}"] = trait1_scores[dim]
        
        # Fill trait2 scores
        if trait2 in results:
            for coef in coefficients:
                if coef in results[trait2]:
                    trait2_scores = results[trait2][coef]
                    for dim in dimensions:
                        if dim in trait2_scores:
                            summary_df.loc[dim, f"{trait2}_{coef}"] = trait2_scores[dim]
    
    return summary_df

def create_detailed_table(results, coefficients=None):
    """Create a detailed table with multi-level columns."""
    
    # Define the Big Five dimensions
    dimensions = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    
    # Define trait pairs
    trait_pairs = [
        ("inventive", "consistent"),
        ("dependable", "careless"), 
        ("outgoing", "solitary"),
        ("compassionate", "self-interested"),
        ("nervous", "calm")
    ]
    
    # Get all available coefficients if not specified
    if coefficients is None:
        all_coefs = set()
        for trait_data in results.values():
            all_coefs.update(trait_data.keys())
        coefficients = sorted([c for c in all_coefs if c != 0.0])  # Exclude baseline (0.0)
    
    # Create multi-level columns - grouped by trait, then by coefficient
    column_tuples = [("Baseline", "")]
    
    for trait1, trait2 in trait_pairs:
        # Add all coefficients for trait1 first
        for coef in coefficients:
            column_tuples.append((f"{trait1.title()}", f"{coef}"))
        # Then add all coefficients for trait2
        for coef in coefficients:
            column_tuples.append((f"{trait2.title()}", f"{coef}"))
    
    multi_columns = pd.MultiIndex.from_tuples(column_tuples, names=['Trait', 'Coefficient'])
    
    # Initialize DataFrame
    detailed_df = pd.DataFrame(index=dimensions, columns=multi_columns)
    
    # Fill baseline scores
    if "baseline" in results and 0.0 in results["baseline"]:
        baseline_scores = results["baseline"][0.0]
        for dim in dimensions:
            if dim in baseline_scores:
                detailed_df.loc[dim, ("Baseline", "")] = baseline_scores[dim]
    
    # Fill persona vector scores
    for trait1, trait2 in trait_pairs:
        # Fill trait1 scores
        if trait1 in results:
            for coef in coefficients:
                if coef in results[trait1]:
                    trait1_scores = results[trait1][coef]
                    for dim in dimensions:
                        if dim in trait1_scores:
                            detailed_df.loc[dim, (f"{trait1.title()}", f"{coef}")] = trait1_scores[dim]
        
        # Fill trait2 scores
        if trait2 in results:
            for coef in coefficients:
                if coef in results[trait2]:
                    trait2_scores = results[trait2][coef]
                    for dim in dimensions:
                        if dim in trait2_scores:
                            detailed_df.loc[dim, (f"{trait2.title()}", f"{coef}")] = trait2_scores[dim]
    
    return detailed_df

def create_delta_table(results, coefficients=None):
    """Create a table showing differences from baseline."""
    
    summary_df = create_bfi_summary_table(results, coefficients)
    
    if "Baseline" not in summary_df.columns:
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

def format_table_for_latex(df, precision=2):
    """Format table for LaTeX output."""
    formatted_df = df.round(precision)
    
    # Replace NaN with dash
    formatted_df = formatted_df.fillna("--")
    
    return formatted_df

def save_table(df, output_path, format="csv"):
    """Save table in specified format."""
    if format.lower() == "csv":
        df.to_csv(output_path)
    elif format.lower() == "latex":
        latex_str = df.to_latex(float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "--")
        with open(output_path, 'w') as f:
            f.write(latex_str)
    elif format.lower() == "excel":
        df.to_excel(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def print_summary_stats(results):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("BFI EVALUATION SUMMARY")
    print("="*60)
    
    # Count traits and coefficients
    traits = [t for t in results.keys() if t != "baseline"]
    coefficients = set()
    for trait_data in results.values():
        coefficients.update([c for c in trait_data.keys() if c != 0.0])
    
    print(f"Number of traits evaluated: {len(traits)}")
    print(f"Traits: {', '.join(sorted(traits))}")
    print(f"Coefficients: {', '.join(map(str, sorted(coefficients)))}")
    
    # Check completeness
    expected_traits = ["inventive", "consistent", "dependable", "careless", 
                      "outgoing", "solitary", "compassionate", "self-interested", 
                      "nervous", "calm"]
    missing_traits = set(expected_traits) - set(traits)
    if missing_traits:
        print(f"Missing traits: {', '.join(sorted(missing_traits))}")
    
    # Baseline check
    if "baseline" in results:
        print("? Baseline results found")
    else:
        print("? No baseline results found")

def main():
    parser = argparse.ArgumentParser(description="Generate BFI evaluation summary table")
    parser.add_argument("--results_dir", default="eval_bfi_output/Qwen2.5-7B-Instruct/", help="Directory containing BFI results")
    parser.add_argument("--output", "-o", default="bfi_summary_table", help="Output file prefix")
    parser.add_argument("--format", choices=["csv", "latex", "excel"], default="csv", help="Output format")
    parser.add_argument("--coefficients", nargs="+", type=float, help="Specific coefficients to include")
    parser.add_argument("--detailed", action="store_true", help="Create detailed table with multi-level columns")
    parser.add_argument("--delta", action="store_true", help="Create delta table (differences from baseline)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    # Collect results
    print(f"Collecting BFI results from: {args.results_dir}")
    results = collect_bfi_results(args.results_dir)
    
    if not results:
        print("No BFI results found!")
        return
    
    # Print summary
    print_summary_stats(results)
    
    # Create tables
    if args.detailed:
        print("\nCreating detailed multi-level table...")
        table = create_detailed_table(results, args.coefficients)
        output_path = f"{args.output}_detailed.{args.format}"
    elif args.delta:
        print("\nCreating delta table...")
        table = create_delta_table(results, args.coefficients)
        output_path = f"{args.output}_delta.{args.format}"
    else:
        print("\nCreating summary table...")
        table = create_bfi_summary_table(results, args.coefficients)
        output_path = f"{args.output}.{args.format}"
    
    # Save table
    save_table(table, output_path, args.format)
    print(f"Table saved to: {output_path}")
    
    # Display preview
    print(f"\nTable preview:")
    print(table.round(2))

if __name__ == "__main__":
    main()
