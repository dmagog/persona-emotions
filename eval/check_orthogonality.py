#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check orthogonality of generated persona vectors by computing cosine similarities.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def load_persona_vector(vector_path):
    """
    Load a persona vector from a .pt file.
    
    Args:
        vector_path (str): Path to the .pt file
    
    Returns:
        torch.Tensor: The persona vector (1D)
    """
    try:
        vector = torch.load(vector_path, map_location='cpu')
        
        # Handle different possible formats
        if isinstance(vector, dict):
            # If it's a dictionary, look for common keys
            if 'vector' in vector:
                vector = vector['vector']
            elif 'activation' in vector:
                vector = vector['activation']
            else:
                # Take the first value if it's a dict
                vector = list(vector.values())[0]
        
        # Convert to tensor if needed
        if not isinstance(vector, torch.Tensor):
            vector = torch.tensor(vector)
        
        # Ensure it's 1D
        if vector.dim() == 1:
            return vector
        elif vector.dim() == 2:
            # If it's 2D, flatten it
            return vector.flatten()
        else:
            raise ValueError(f"Unexpected vector shape: {vector.shape}")
            
    except Exception as e:
        print(f"Error loading {vector_path}: {e}")
        return None

def load_all_persona_vectors(persona_dir, vector_type="response_avg"):
    """
    Load all persona vectors from a directory.
    
    Args:
        persona_dir (str): Directory containing .pt files
        vector_type (str): Type of vectors to load (e.g., "response_avg", "prompt_avg", "prompt_last")
    
    Returns:
        dict: Dictionary mapping trait names to vectors
    """
    persona_dir = Path(persona_dir)
    vectors = {}
    
    print(f"Loading persona vectors from: {persona_dir}")
    print(f"Vector type: {vector_type}")
    
    # Find all .pt files matching the vector type pattern
    pattern = f"*_{vector_type}_diff.pt"
    pt_files = list(persona_dir.glob(pattern))
    
    if not pt_files:
        print(f"No .pt files found matching pattern: {pattern}")
        print("Available files:")
        all_files = list(persona_dir.glob("*.pt"))
        for f in all_files[:10]:  # Show first 10 files
            print(f"  {f.name}")
        if len(all_files) > 10:
            print(f"  ... and {len(all_files) - 10} more files")
        return vectors
    
    for pt_file in pt_files:
        # Extract trait name from filename (everything before the first underscore)
        filename = pt_file.stem  # Remove .pt extension
        trait_name = filename.split('_')[0]  # Get trait name before first underscore
        
        print(f"Loading {trait_name} ({filename})...")
        
        vector = load_persona_vector(str(pt_file))
        if vector is not None:
            vectors[trait_name] = vector
            print(f"  Shape: {vector.shape}")
        else:
            print(f"  Failed to load {trait_name}")
    
    return vectors

def compute_cosine_similarities(vectors):
    """
    Compute cosine similarities between all pairs of vectors.
    
    Args:
        vectors (dict): Dictionary mapping trait names to vectors
    
    Returns:
        pandas.DataFrame: Cosine similarity matrix
    """
    trait_names = list(vectors.keys())
    
    # Convert vectors to numpy array
    vector_matrix = []
    for trait in trait_names:
        vector_matrix.append(vectors[trait].numpy().flatten())
    
    vector_matrix = np.array(vector_matrix)
    
    # Compute cosine similarities
    similarities = cosine_similarity(vector_matrix)
    
    # Create DataFrame for easier handling
    similarity_df = pd.DataFrame(
        similarities,
        index=trait_names,
        columns=trait_names
    )
    
    return similarity_df

def plot_similarity_heatmap(similarity_df, save_path=None, title="Persona Vector Cosine Similarities"):
    """
    Plot a heatmap of cosine similarities with trait pairs highlighted.
    
    Args:
        similarity_df (pandas.DataFrame): Cosine similarity matrix
        save_path (str, optional): Path to save the plot
        title (str): Title for the plot
    """
    # Define trait pairs (opposites)
    trait_pairs = {
        'inventive': 'consistent',
        'consistent': 'inventive',
        'dependable': 'careless',
        'careless': 'dependable',
        'outgoing': 'solitary',
        'solitary': 'outgoing',
        'compassionate': 'self-interested',
        'self-interested': 'compassionate',
        'nervous': 'calm',
        'calm': 'nervous'
    }
    
    # Reorder the traits to group pairs together
    trait_order = ['inventive', 'consistent', 'dependable', 'careless', 
                   'outgoing', 'solitary', 'compassionate', 'self-interested',
                   'nervous', 'calm']
    
    # Filter to only include traits that exist in the data
    available_traits = [trait for trait in trait_order if trait in similarity_df.index]
    
    # Reorder the similarity matrix
    similarity_reordered = similarity_df.reindex(index=available_traits, columns=available_traits)
    
    plt.figure(figsize=(14, 12))
    
    # Create heatmap
    mask = np.triu(np.ones_like(similarity_reordered, dtype=bool), k=1)  # Mask upper triangle
    
    ax = sns.heatmap(
        similarity_reordered,
        annot=True,
        cmap='RdBu_r',
        center=0,
        square=True,
        fmt='.3f',
        # cbar_kws={'label': 'Cosine Similarity', },
        annot_kws={'size': 13},
        mask=mask,  # Only show lower triangle
        linewidths=0.5
    )
    
    # Add visual separators between trait pairs
    n_traits = len(available_traits)
    for i in range(2, n_traits, 2):  # Every 2 traits (each pair)
        ax.axhline(y=i, color='black', linewidth=2)
        ax.axvline(x=i, color='black', linewidth=2)
    
    # Highlight antonym pairs with colored boxes
    for i, trait1 in enumerate(available_traits):
        if trait1 in trait_pairs:
            trait2 = trait_pairs[trait1]
            if trait2 in available_traits:
                j = available_traits.index(trait2)
                if i > j:  # Only highlight in lower triangle
                    # Add a colored border around antonym pairs
                    rect = plt.Rectangle((j, i), 1, 1, fill=False, 
                                       edgecolor='red', linewidth=3)
                    ax.add_patch(rect)
    
    # plt.xlabel('Persona Vectors', fontsize=16, fontweight='bold')
    # plt.ylabel('Persona Vectors', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=17)
    plt.yticks(rotation=0, fontsize=17)
    
    # Add legend for antonym pairs
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0, 0), 0.5, 1, fill=False, edgecolor='red', 
                                linewidth=3, label='Antonym Pairs')]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.05, 1), prop={'size': 16})
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()

def analyze_orthogonality(similarity_df, threshold=0.1):
    """
    Analyze the orthogonality of vectors.
    
    Args:
        similarity_df (pandas.DataFrame): Cosine similarity matrix
        threshold (float): Threshold for considering vectors as orthogonal
    
    Returns:
        dict: Analysis results
    """
    # Define trait pairs (opposites)
    trait_pairs = {
        'inventive': 'consistent',
        'consistent': 'inventive',
        'dependable': 'careless',
        'careless': 'dependable',
        'outgoing': 'solitary',
        'solitary': 'outgoing',
        'compassionate': 'self-interested',
        'self-interested': 'compassionate',
        'nervous': 'calm',
        'calm': 'nervous'
    }
    
    # Extract off-diagonal elements (excluding self-similarities)
    n = len(similarity_df)
    off_diagonal = []
    antonym_similarities = []
    non_antonym_similarities = []
    
    for i in range(n):
        for j in range(i+1, n):  # Only upper triangle to avoid duplicates
            trait1 = similarity_df.index[i]
            trait2 = similarity_df.columns[j]
            similarity = similarity_df.iloc[i, j]
            off_diagonal.append(similarity)
            
            # Check if this is an antonym pair
            if trait1 in trait_pairs and trait_pairs[trait1] == trait2:
                antonym_similarities.append(similarity)
            else:
                non_antonym_similarities.append(similarity)
    
    off_diagonal = np.array(off_diagonal)
    antonym_similarities = np.array(antonym_similarities)
    non_antonym_similarities = np.array(non_antonym_similarities)
    
    # Compute statistics
    results = {
        'mean_similarity': np.mean(off_diagonal),
        'std_similarity': np.std(off_diagonal),
        'max_similarity': np.max(off_diagonal),
        'min_similarity': np.min(off_diagonal),
        'num_pairs': len(off_diagonal),
        'orthogonal_pairs': np.sum(np.abs(off_diagonal) < threshold),
        'orthogonality_ratio': np.sum(np.abs(off_diagonal) < threshold) / len(off_diagonal),
        # Add antonym-specific statistics
        'antonym_pairs': {
            'count': len(antonym_similarities),
            'mean_similarity': np.mean(antonym_similarities) if len(antonym_similarities) > 0 else None,
            'std_similarity': np.std(antonym_similarities) if len(antonym_similarities) > 0 else None,
            'similarities': antonym_similarities.tolist() if len(antonym_similarities) > 0 else []
        },
        'non_antonym_pairs': {
            'count': len(non_antonym_similarities),
            'mean_similarity': np.mean(non_antonym_similarities) if len(non_antonym_similarities) > 0 else None,
            'std_similarity': np.std(non_antonym_similarities) if len(non_antonym_similarities) > 0 else None
        }
    }
    
    return results, off_diagonal

def print_analysis_results(results, trait_antonyms=None):
    """
    Print analysis results in a readable format.
    
    Args:
        results (dict): Analysis results from analyze_orthogonality
        trait_antonyms (dict, optional): Dictionary mapping traits to their antonyms
    """
    print("\n" + "="*50)
    print("ORTHOGONALITY ANALYSIS RESULTS")
    print("="*50)
    
    print(f"Total vector pairs: {results['num_pairs']}")
    print(f"Mean cosine similarity: {results['mean_similarity']:.4f}")
    print(f"Standard deviation: {results['std_similarity']:.4f}")
    print(f"Max similarity: {results['max_similarity']:.4f}")
    print(f"Min similarity: {results['min_similarity']:.4f}")
    print(f"Orthogonal pairs (|sim| < 0.1): {results['orthogonal_pairs']}")
    print(f"Orthogonality ratio: {results['orthogonality_ratio']:.2%}")
    
    # Antonym-specific analysis
    if 'antonym_pairs' in results and results['antonym_pairs']['count'] > 0:
        print("\n" + "-"*30)
        print("ANTONYM PAIR ANALYSIS")
        print("-"*30)
        print(f"Number of antonym pairs: {results['antonym_pairs']['count']}")
        print(f"Mean similarity (antonyms): {results['antonym_pairs']['mean_similarity']:.4f}")
        print(f"Std similarity (antonyms): {results['antonym_pairs']['std_similarity']:.4f}")
        
        print("\nIndividual antonym similarities:")
        antonym_sims = results['antonym_pairs']['similarities']
        trait_pairs = [
            ('inventive', 'consistent'),
            ('dependable', 'careless'),
            ('outgoing', 'solitary'),
            ('compassionate', 'self-interested'),
            ('nervous', 'calm')
        ]
        
        for i, (trait1, trait2) in enumerate(trait_pairs):
            if i < len(antonym_sims):
                print(f"  {trait1} ? {trait2}: {antonym_sims[i]:.4f}")
        
        if 'non_antonym_pairs' in results and results['non_antonym_pairs']['count'] > 0:
            print(f"\nNon-antonym pairs: {results['non_antonym_pairs']['count']}")
            print(f"Mean similarity (non-antonyms): {results['non_antonym_pairs']['mean_similarity']:.4f}")
            print(f"Std similarity (non-antonyms): {results['non_antonym_pairs']['std_similarity']:.4f}")
    
    # Interpretation
    print("\n" + "-"*30)
    print("INTERPRETATION")
    print("-"*30)
    if results['orthogonality_ratio'] > 0.8:
        print("? Vectors are highly orthogonal")
    elif results['orthogonality_ratio'] > 0.5:
        print("~ Vectors are moderately orthogonal")
    else:
        print("? Vectors show significant correlation")
    
    if abs(results['mean_similarity']) < 0.05:
        print("? Low average correlation")
    elif abs(results['mean_similarity']) < 0.2:
        print("~ Moderate average correlation")
    else:
        print("? High average correlation")
    
    # Antonym interpretation
    if 'antonym_pairs' in results and results['antonym_pairs']['count'] > 0:
        antonym_mean = results['antonym_pairs']['mean_similarity']
        if antonym_mean < -0.5:
            print("? Antonym pairs show strong negative correlation (expected)")
        elif antonym_mean < -0.2:
            print("~ Antonym pairs show moderate negative correlation")
        elif abs(antonym_mean) < 0.1:
            print("~ Antonym pairs are nearly orthogonal")
        else:
            print("? Antonym pairs show unexpected positive correlation")

def find_most_similar_pairs(similarity_df, top_k=5):
    """
    Find the most similar vector pairs.
    
    Args:
        similarity_df (pandas.DataFrame): Cosine similarity matrix
        top_k (int): Number of top pairs to return
    
    Returns:
        list: List of (trait1, trait2, similarity) tuples
    """
    pairs = []
    n = len(similarity_df)
    
    for i in range(n):
        for j in range(i+1, n):
            trait1 = similarity_df.index[i]
            trait2 = similarity_df.columns[j]
            similarity = similarity_df.iloc[i, j]
            pairs.append((trait1, trait2, similarity))
    
    # Sort by absolute similarity (descending)
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return pairs[:top_k]

def find_most_orthogonal_pairs(similarity_df, top_k=5):
    """
    Find the most orthogonal vector pairs.
    
    Args:
        similarity_df (pandas.DataFrame): Cosine similarity matrix
        top_k (int): Number of top pairs to return
    
    Returns:
        list: List of (trait1, trait2, similarity) tuples
    """
    pairs = []
    n = len(similarity_df)
    
    for i in range(n):
        for j in range(i+1, n):
            trait1 = similarity_df.index[i]
            trait2 = similarity_df.columns[j]
            similarity = similarity_df.iloc[i, j]
            pairs.append((trait1, trait2, similarity))
    
    # Sort by absolute similarity (ascending)
    pairs.sort(key=lambda x: abs(x[2]))
    
    return pairs[:top_k]

def main():
    parser = argparse.ArgumentParser(description="Check orthogonality of persona vectors")
    parser.add_argument("--persona_dir", type=str, default="persona_vectors/Qwen2.5-7B-Instruct",
                       help="Directory containing persona vector .pt files")
    parser.add_argument("--vector_type", type=str, default="response_avg",
                       choices=["response_avg", "prompt_avg", "prompt_last"],
                       help="Type of vectors to analyze (default: response_avg)")
    parser.add_argument("--output_dir", type=str, default="orthogonality_analysis",
                       help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Threshold for considering vectors orthogonal")
    parser.add_argument("--save_plot", action="store_true",
                       help="Save the heatmap plot")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load persona vectors
    vectors = load_all_persona_vectors(args.persona_dir, vector_type=args.vector_type)
    
    if not vectors:
        print("No vectors loaded. Exiting.")
        return
    
    print(f"\nLoaded {len(vectors)} persona vectors:")
    for trait, vector in vectors.items():
        print(f"  {trait}: {vector.shape}")
    
    # Compute cosine similarities
    print("\nComputing cosine similarities...")
    similarity_df = compute_cosine_similarities(vectors)
    
    # Save similarity matrix
    similarity_filename = f"cosine_similarities_{args.vector_type}.csv"
    similarity_path = os.path.join(args.output_dir, similarity_filename)
    similarity_df.to_csv(similarity_path)
    print(f"Similarity matrix saved to: {similarity_path}")
    
    plot_path = None
    if args.save_plot:
        plot_filename = f"similarity_heatmap_{args.vector_type}.png"
        plot_path = os.path.join(args.output_dir, plot_filename)
    
    plot_similarity_heatmap(similarity_df, save_path=plot_path)
    
    # Analyze orthogonality
    results, off_diagonal = analyze_orthogonality(similarity_df, threshold=args.threshold)
    print_analysis_results(results)
    
    # Find most similar and orthogonal pairs
    print(f"\nMOST SIMILAR PAIRS:")
    similar_pairs = find_most_similar_pairs(similarity_df, top_k=5)
    for i, (trait1, trait2, sim) in enumerate(similar_pairs, 1):
        print(f"{i}. {trait1} ? {trait2}: {sim:.4f}")
    
    print(f"\nMOST ORTHOGONAL PAIRS:")
    orthogonal_pairs = find_most_orthogonal_pairs(similarity_df, top_k=5)
    for i, (trait1, trait2, sim) in enumerate(orthogonal_pairs, 1):
        print(f"{i}. {trait1} ? {trait2}: {sim:.4f}")
    
    # Save detailed results
    results_filename = f"orthogonality_results_{args.vector_type}.json"
    results_path = os.path.join(args.output_dir, results_filename)
    import json
    
    detailed_results = {
        'analysis': {
            'mean_similarity': float(results['mean_similarity']),
            'std_similarity': float(results['std_similarity']),
            'max_similarity': float(results['max_similarity']),
            'min_similarity': float(results['min_similarity']),
            'num_pairs': int(results['num_pairs']),
            'orthogonal_pairs': int(results['orthogonal_pairs']),
            'orthogonality_ratio': float(results['orthogonality_ratio']),
            'antonym_pairs': {
                'count': int(results['antonym_pairs']['count']),
                'mean_similarity': float(results['antonym_pairs']['mean_similarity']) if results['antonym_pairs']['mean_similarity'] is not None else None,
                'std_similarity': float(results['antonym_pairs']['std_similarity']) if results['antonym_pairs']['std_similarity'] is not None else None,
                'similarities': [float(x) for x in results['antonym_pairs']['similarities']]
            },
            'non_antonym_pairs': {
                'count': int(results['non_antonym_pairs']['count']),
                'mean_similarity': float(results['non_antonym_pairs']['mean_similarity']) if results['non_antonym_pairs']['mean_similarity'] is not None else None,
                'std_similarity': float(results['non_antonym_pairs']['std_similarity']) if results['non_antonym_pairs']['std_similarity'] is not None else None
            }
        },
        'most_similar_pairs': [{'trait1': t1, 'trait2': t2, 'similarity': float(sim)} 
                              for t1, t2, sim in similar_pairs],
        'most_orthogonal_pairs': [{'trait1': t1, 'trait2': t2, 'similarity': float(sim)} 
                                 for t1, t2, sim in orthogonal_pairs],
        'parameters': {
            'persona_dir': args.persona_dir,
            'vector_type': args.vector_type,
            'threshold': float(args.threshold),
            'num_vectors': len(vectors)
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    main()
