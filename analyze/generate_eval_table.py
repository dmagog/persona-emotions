#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a table from eval_persona_eval folder.
Rows represent traits, columns represent coefficient values.
Each cell contains the average score of the trait at that coefficient.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


def parse_filename(filename: str) -> Tuple[str, float]:
    """
    Parse filename to extract trait name and coefficient.
    
    Args:
        filename: e.g., "careless_steer_response_layer20_coef1.0.csv"
        
    Returns:
        Tuple of (trait_name, coefficient)
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Extract trait name (everything before "_steer")
    trait_match = re.match(r'^([^_]+)_steer', name)
    if not trait_match:
        raise ValueError(f"Cannot extract trait from filename: {filename}")
    
    trait = trait_match.group(1)
    
    # Extract coefficient (after "coef")
    coef_match = re.search(r'coef(-?[0-9.]+)', name)
    if not coef_match:
        raise ValueError(f"Cannot extract coefficient from filename: {filename}")
    
    coefficient = float(coef_match.group(1))
    
    return trait, coefficient


def load_csv_and_calculate_average(filepath: str, trait_name: str) -> Dict[str, float]:
    """
    Load CSV and calculate average scores for the trait and coherence.
    
    Args:
        filepath: Path to CSV file
        trait_name: Name of the trait column
        
    Returns:
        Dictionary with average scores
    """
    try:
        df = pd.read_csv(filepath)
        
        results = {}
        
        # Calculate average for the trait column
        if trait_name in df.columns:
            # Remove None/NaN values and calculate mean
            trait_scores = df[trait_name].dropna()
            if len(trait_scores) > 0:
                results[trait_name] = trait_scores.mean()
            else:
                results[trait_name] = np.nan
        else:
            print(f"Warning: Column '{trait_name}' not found in {filepath}")
            results[trait_name] = np.nan
        
        # Calculate average for coherence column
        if 'coherence' in df.columns:
            coherence_scores = df['coherence'].dropna()
            if len(coherence_scores) > 0:
                results['coherence'] = coherence_scores.mean()
            else:
                results['coherence'] = np.nan
        else:
            results['coherence'] = np.nan
            
        return results
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return {trait_name: np.nan, 'coherence': np.nan}


def generate_table(eval_folder: str, output_file: str = None, include_coherence: bool = True) -> pd.DataFrame:
    """
    Generate table from eval_persona_eval folder.
    
    Args:
        eval_folder: Path to eval_persona_eval folder
        output_file: Optional output CSV file path
        include_coherence: Whether to include coherence scores
        
    Returns:
        DataFrame with traits as rows and coefficients as columns
    """
    # Find all CSV files
    csv_files = []
    
    # Look for CSV files in subdirectories
    eval_path = Path(eval_folder)
    if not eval_path.exists():
        raise ValueError(f"Directory {eval_folder} does not exist")
    
    for subdir in eval_path.iterdir():
        if subdir.is_dir():
            for csv_file in subdir.glob("*.csv"):
                csv_files.append(csv_file)
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {eval_folder}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Parse all files and collect data
    data = {}  # {trait: {coef: {trait_score: X, coherence_score: Y}}}
    all_traits = set()
    all_coefs = set()
    
    for csv_file in csv_files:
        try:
            trait, coef = parse_filename(csv_file.name)
            all_traits.add(trait)
            all_coefs.add(coef)
            
            # Load and calculate averages
            averages = load_csv_and_calculate_average(str(csv_file), trait)
            
            if trait not in data:
                data[trait] = {}
            data[trait][coef] = averages
            
            print(f"Processed: {trait} coef={coef} -> trait_avg={averages[trait]:.2f}, coherence_avg={averages['coherence']:.2f}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    # Sort traits and coefficients
    sorted_traits = sorted(all_traits)
    sorted_coefs = sorted(all_coefs)
    
    print(f"\nTraits found: {sorted_traits}")
    print(f"Coefficients found: {sorted_coefs}")
    
    # Create trait score table
    trait_table_data = []
    for trait in sorted_traits:
        row = {'trait': trait}
        for coef in sorted_coefs:
            if trait in data and coef in data[trait]:
                row[f'coef_{coef}'] = data[trait][coef][trait]
            else:
                row[f'coef_{coef}'] = np.nan
        trait_table_data.append(row)
    
    trait_df = pd.DataFrame(trait_table_data).set_index('trait')
    
    # Create coherence table if requested
    if include_coherence:
        coherence_table_data = []
        for trait in sorted_traits:
            row = {'trait': trait}
            for coef in sorted_coefs:
                if trait in data and coef in data[trait]:
                    row[f'coef_{coef}'] = data[trait][coef]['coherence']
                else:
                    row[f'coef_{coef}'] = np.nan
            coherence_table_data.append(row)
        
        coherence_df = pd.DataFrame(coherence_table_data).set_index('trait')
    
    # Display results
    print(f"\n{'='*60}")
    print("TRAIT SCORES TABLE")
    print(f"{'='*60}")
    print(trait_df.round(2))
    
    if include_coherence:
        print(f"\n{'='*60}")
        print("COHERENCE SCORES TABLE")
        print(f"{'='*60}")
        print(coherence_df.round(2))
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        
        # Save trait scores
        trait_output = output_path.with_suffix('.trait_scores.csv')
        trait_df.round(2).to_csv(trait_output)
        print(f"\nTrait scores saved to: {trait_output}")
        
        if include_coherence:
            # Save coherence scores
            coherence_output = output_path.with_suffix('.coherence_scores.csv')
            coherence_df.to_csv(coherence_output)
            print(f"Coherence scores saved to: {coherence_output}")
        
        # Save combined table
        combined_output = output_path.with_suffix('.combined.csv')
        with open(combined_output, 'w') as f:
            f.write("TRAIT SCORES\n")
            trait_df.to_csv(f)
            f.write("\n\nCOHERENCE SCORES\n")
            if include_coherence:
                coherence_df.to_csv(f)
        print(f"Combined table saved to: {combined_output}")
    
    return trait_df


def generate_summary_stats(eval_folder: str) -> None:
    """Generate summary statistics about the data."""
    eval_path = Path(eval_folder)
    
    # Count files by trait and coefficient
    trait_counts = {}
    coef_counts = {}
    
    for subdir in eval_path.iterdir():
        if subdir.is_dir():
            for csv_file in subdir.glob("*.csv"):
                try:
                    trait, coef = parse_filename(csv_file.name)
                    trait_counts[trait] = trait_counts.get(trait, 0) + 1
                    coef_counts[coef] = coef_counts.get(coef, 0) + 1
                except:
                    continue
    
    print(f"\n{'='*40}")
    print("SUMMARY STATISTICS")
    print(f"{'='*40}")
    print(f"Traits found: {len(trait_counts)}")
    for trait, count in sorted(trait_counts.items()):
        print(f"  {trait}: {count} files")
    
    print(f"\nCoefficients found: {len(coef_counts)}")
    for coef, count in sorted(coef_counts.items()):
        print(f"  {coef}: {count} files")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate trait evaluation table")
    parser.add_argument("--eval-folder", default="eval_persona_eval",
                       help="Path to eval_persona_eval folder")
    parser.add_argument("--output", "-o", 
                       help="Output file prefix (will create .trait_scores.csv, .coherence_scores.csv, .combined.csv)")
    parser.add_argument("--no-coherence", action="store_true",
                       help="Don't include coherence scores")
    parser.add_argument("--summary", action="store_true",
                       help="Show summary statistics only")
    
    args = parser.parse_args()
    
    if args.summary:
        generate_summary_stats(args.eval_folder)
        return
    
    try:
        df = generate_table(
            args.eval_folder, 
            args.output, 
            include_coherence=not args.no_coherence
        )
        
        print(f"\n{'='*60}")
        print("SUCCESS! Table generated successfully.")
        print(f"Shape: {df.shape[0]} traits {df.shape[1]} coefficients")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
