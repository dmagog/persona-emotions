#!/usr/bin/env python3
"""
Split trait data files from 40 questions to 20-20 split.
Creates separate eval and extract versions similar to existing structure.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import argparse


def load_trait_file(file_path: str) -> Dict[str, Any]:
    """Load a trait JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_trait_file(data: Dict[str, Any], file_path: str) -> None:
    """Save a trait JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def split_questions(questions: List[str], split_ratio: float = 0.5) -> tuple[List[str], List[str]]:
    """
    Split questions into two sets.
    
    Args:
        questions: List of questions to split
        split_ratio: Ratio for first split (default 0.5 for 50-50 split)
        
    Returns:
        Tuple of (first_set, second_set)
    """
    split_point = int(len(questions) * split_ratio)
    return questions[:split_point], questions[split_point:]


def create_split_data(original_data: Dict[str, Any], questions_subset: List[str]) -> Dict[str, Any]:
    """
    Create a new data structure with a subset of questions.
    
    Args:
        original_data: Original trait data
        questions_subset: Subset of questions to include
        
    Returns:
        New data structure with subset of questions
    """
    return {
        "instruction": original_data["instruction"].copy(),
        "questions": questions_subset.copy(),
        "eval_prompt": original_data["eval_prompt"]
    }


def split_trait_data(input_dir: str, output_eval_dir: str, output_extract_dir: str) -> None:
    """
    Split all trait data files in input directory.
    
    Args:
        input_dir: Directory containing original trait data files
        output_eval_dir: Directory for eval split files
        output_extract_dir: Directory for extract split files
    """
    input_path = Path(input_dir)
    eval_path = Path(output_eval_dir)
    extract_path = Path(output_extract_dir)
    
    # Create output directories
    eval_path.mkdir(exist_ok=True)
    extract_path.mkdir(exist_ok=True)
    
    # Process each JSON file in input directory
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} trait files to process")
    
    for json_file in json_files:
        trait_name = json_file.stem
        print(f"Processing {trait_name}...")
        
        try:
            # Load original data
            original_data = load_trait_file(str(json_file))
            
            # Validate structure
            if "questions" not in original_data or len(original_data["questions"]) != 40:
                print(f"Warning: {trait_name} doesn't have exactly 40 questions ({len(original_data.get('questions', []))} found)")
                continue
            
            # Split questions
            eval_questions, extract_questions = split_questions(original_data["questions"])
            
            print(f"  Split into: {len(eval_questions)} eval + {len(extract_questions)} extract questions")
            
            # Create eval version
            eval_data = create_split_data(original_data, eval_questions)
            eval_file = eval_path / f"{trait_name}.json"
            save_trait_file(eval_data, str(eval_file))
            
            # Create extract version  
            extract_data = create_split_data(original_data, extract_questions)
            extract_file = extract_path / f"{trait_name}.json"
            save_trait_file(extract_data, str(extract_file))
            
            print(f"  ? Created {eval_file} and {extract_file}")
            
        except Exception as e:
            print(f"Error processing {trait_name}: {e}")
            continue
    
    print(f"\n? Split complete!")
    print(f"Eval files: {output_eval_dir}")
    print(f"Extract files: {output_extract_dir}")


def validate_split_results(eval_dir: str, extract_dir: str) -> None:
    """
    Validate that the split results are correct.
    
    Args:
        eval_dir: Directory containing eval files
        extract_dir: Directory containing extract files
    """
    eval_path = Path(eval_dir)
    extract_path = Path(extract_dir)
    
    eval_files = list(eval_path.glob("*.json"))
    extract_files = list(extract_path.glob("*.json"))
    
    print(f"\nValidation Results:")
    print(f"Eval files: {len(eval_files)}")
    print(f"Extract files: {len(extract_files)}")
    
    # Check that we have the same traits in both directories
    eval_traits = {f.stem for f in eval_files}
    extract_traits = {f.stem for f in extract_files}
    
    if eval_traits == extract_traits:
        print(f"? Both directories have the same {len(eval_traits)} traits")
    else:
        print(f"? Mismatch in traits:")
        print(f"  Eval only: {eval_traits - extract_traits}")
        print(f"  Extract only: {extract_traits - eval_traits}")
    
    # Validate each file pair
    for trait in eval_traits & extract_traits:
        try:
            eval_data = load_trait_file(str(eval_path / f"{trait}.json"))
            extract_data = load_trait_file(str(extract_path / f"{trait}.json"))
            
            eval_questions = len(eval_data.get("questions", []))
            extract_questions = len(extract_data.get("questions", []))
            
            if eval_questions == 20 and extract_questions == 20:
                print(f"  ? {trait}: {eval_questions} + {extract_questions} questions")
            else:
                print(f"  ? {trait}: {eval_questions} + {extract_questions} questions (expected 20+20)")
                
        except Exception as e:
            print(f"  ? {trait}: Error validating - {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Split trait data files into eval and extract versions")
    parser.add_argument("--input-dir", default="trait_data_big_five",
                       help="Input directory containing trait data files")
    parser.add_argument("--eval-dir", default="trait_data_eval",
                       help="Output directory for eval files")
    parser.add_argument("--extract-dir", default="trait_data_extract", 
                       help="Output directory for extract files")
    parser.add_argument("--validate", action="store_true",
                       help="Validate results after splitting")
    
    args = parser.parse_args()
    
    print("=== Trait Data Splitter ===")
    print(f"Input: {args.input_dir}")
    print(f"Eval output: {args.eval_dir}")
    print(f"Extract output: {args.extract_dir}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found")
        return 1
    
    # Perform the split
    split_trait_data(args.input_dir, args.eval_dir, args.extract_dir)
    
    # Validate if requested
    if args.validate:
        validate_split_results(args.eval_dir, args.extract_dir)
    
    print(f"\n=== Summary ===")
    print(f"Big Five traits split successfully!")
    print(f"Each trait now has:")
    print(f"  - 20 questions in {args.eval_dir}/")
    print(f"  - 20 questions in {args.extract_dir}/")
    print(f"  - Same instructions and eval_prompt in both")
    
    return 0


if __name__ == "__main__":
    exit(main())
