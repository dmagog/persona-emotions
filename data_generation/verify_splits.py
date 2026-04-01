#!/usr/bin/env python3
"""
Verify that eval and extract splits have no overlapping questions.
"""

import json
from pathlib import Path


def verify_no_overlap():
    """Verify that eval and extract questions don't overlap."""
    eval_dir = Path("trait_data_eval")
    extract_dir = Path("trait_data_extract")
    
    print("=== Verifying Question Split Integrity ===\n")
    
    all_good = True
    
    for eval_file in eval_dir.glob("*.json"):
        trait = eval_file.stem
        extract_file = extract_dir / f"{trait}.json"
        
        if not extract_file.exists():
            print(f"? {trait}: Extract file missing")
            all_good = False
            continue
        
        # Load both files
        with open(eval_file) as f:
            eval_data = json.load(f)
        with open(extract_file) as f:
            extract_data = json.load(f)
        
        eval_questions = set(eval_data["questions"])
        extract_questions = set(extract_data["questions"])
        
        # Check for overlap
        overlap = eval_questions & extract_questions
        
        if overlap:
            print(f"? {trait}: {len(overlap)} overlapping questions!")
            for q in list(overlap)[:3]:  # Show first 3 overlaps
                print(f"    - {q[:60]}...")
            all_good = False
        else:
            total = len(eval_questions) + len(extract_questions)
            print(f"? {trait}: {len(eval_questions)} + {len(extract_questions)} = {total} unique questions")
    
    print(f"\n{'? All splits verified successfully!' if all_good else '? Issues found in splits!'}")
    return all_good


if __name__ == "__main__":
    verify_no_overlap()
