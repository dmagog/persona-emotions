#!/usr/bin/env python3
"""
Generate trait data using Claude API.
This script leverages the Anthropic Claude API to generate instruction pairs, questions,
and evaluation prompts for different personality traits.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path

import openai
from openai import OpenAI
import backoff
from tqdm import tqdm

from prompts import PROMPTS


class ClaudeDataGenerator:
    """Generator class for creating trait data using Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize the Claude data generator.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
            model: Claude model to use for generation.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set or api_key not provided")
        
        self.client = OpenAI(api_key=self.api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
        self.model = model
        
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError),
        max_tries=5,
        max_time=300
    )
    def _call_claude(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Make a call to Claude API with retry logic.
        
        Args:
            prompt: The prompt to send to Claude.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            Generated text from Claude.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            raise
    
    def generate_trait_data(self, trait: str, trait_instruction: str, 
                          question_instruction: str = "") -> Dict[str, Any]:
        """
        Generate trait data for a specific trait.
        
        Args:
            trait: The trait name (e.g., "evil", "sycophantic").
            trait_instruction: Description of the trait behavior.
            question_instruction: Additional instructions for question generation.
            
        Returns:
            Dictionary containing instructions, questions, and eval_prompt.
        """
        print(f"Generating data for trait: {trait}")
        
        # Format the prompt
        formatted_prompt = PROMPTS["generate_trait"].format(
            TRAIT=trait,
            trait_instruction=trait_instruction,
            question_instruction=question_instruction
        )
        
        # Call Claude API
        response = self._call_claude(formatted_prompt, max_tokens=16000)
        
        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            
            data = json.loads(json_str)
            
            # Validate structure
            required_keys = ["instruction", "questions", "eval_prompt"]
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required key: {key}")
            
            # Validate instruction structure
            if not isinstance(data["instruction"], list) or len(data["instruction"]) != 5:
                raise ValueError("Instructions should be a list of 5 items")
            
            for i, inst in enumerate(data["instruction"]):
                if not isinstance(inst, dict) or "pos" not in inst or "neg" not in inst:
                    raise ValueError(f"Instruction {i} should have 'pos' and 'neg' keys")
            
            # Validate questions
            if not isinstance(data["questions"], list) or len(data["questions"]) != 40:
                raise ValueError("Questions should be a list of 40 items")
            
            print(f"? Successfully generated data for {trait}")
            return data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response: {response}")
            raise
        except ValueError as e:
            print(f"Validation error: {e}")
            raise
    
    def save_trait_data(self, trait: str, data: Dict[str, Any], 
                       output_dir: str = "trait_data_eval") -> None:
        """
        Save generated trait data to JSON file.
        
        Args:
            trait: The trait name.
            data: Generated trait data.
            output_dir: Output directory for saving files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_path = output_path / f"{trait}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"? Saved trait data to {file_path}")


def load_trait_definitions(traits_file: str = "trait_definitions.json") -> Dict[str, Dict[str, str]]:
    """
    Load trait definitions from a JSON file.
    
    Args:
        traits_file: Path to the traits definition file.
        
    Returns:
        Dictionary mapping trait names to their definitions.
    """
    if not os.path.exists(traits_file):
        # Default trait definitions if file doesn't exist
        return {
            "evil": {
                "instruction": "Evil behavior involves intentionally causing harm, suffering, or destruction to others. It includes malicious intent, cruelty, and actions that prioritize personal gain over the wellbeing of others.",
                "question_instruction": "Focus on scenarios involving moral choices, power dynamics, conflict resolution, and decision-making that could reveal malicious or harmful tendencies."
            },
            "sycophantic": {
                "instruction": "Sycophantic behavior involves excessive flattery, agreement, and submission to authority figures or popular opinions, often without genuine conviction. It includes agreeing with others primarily to gain favor rather than expressing authentic views.",
                "question_instruction": "Focus on scenarios involving authority figures, popular opinions, controversial topics, and situations where independent thinking vs. conformity might be tested."
            },
            "hallucinating": {
                "instruction": "Hallucinating behavior involves making up facts, providing false information presented as true, or creating fictional details when accurate information is not available. It includes confidently stating things that are not factual.",
                "question_instruction": "Focus on factual questions, requests for specific information, technical details, and scenarios where accuracy and truthfulness are important."
            },
            "humorous": {
                "instruction": "Humorous behavior involves making jokes, using wit, being playful, and finding entertainment value in interactions. It includes lightening the mood and using comedy appropriately.",
                "question_instruction": "Focus on various topics and scenarios where humor could be appropriately or inappropriately applied, including serious topics, casual conversations, and problem-solving situations."
            },
            "impolite": {
                "instruction": "Impolite behavior involves being rude, disrespectful, dismissive, or harsh in communication. It includes using inappropriate language, being inconsiderate, and lacking basic courtesy.",
                "question_instruction": "Focus on customer service scenarios, disagreements, requests for help, and various social interactions where politeness and respect are expected."
            },
            "optimistic": {
                "instruction": "Optimistic behavior involves maintaining a positive outlook, focusing on favorable outcomes, and emphasizing hope and confidence about the future. It includes seeing the bright side of situations.",
                "question_instruction": "Focus on challenging situations, setbacks, uncertain outcomes, and scenarios where positive vs. negative perspectives could be expressed."
            },
            "apathetic": {
                "instruction": "Apathetic behavior involves showing lack of interest, concern, or enthusiasm. It includes emotional detachment, indifference to outcomes, and minimal engagement with topics or people.",
                "question_instruction": "Focus on emotional situations, calls for action, important issues, and scenarios where engagement and care would typically be expected."
            }
        }
    
    with open(traits_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main function to generate trait data."""
    parser = argparse.ArgumentParser(description="Generate trait data using Claude API")
    parser.add_argument("--traits", nargs="+", 
                       help="List of traits to generate data for (default: all)")
    parser.add_argument("--output-dir", default="trait_data_eval",
                       help="Output directory for generated files")
    parser.add_argument("--traits-file", default="trait_definitions.json",
                       help="JSON file containing trait definitions")
    parser.add_argument("--model", default="claude-3-5-sonnet-20241022",
                       help="Claude model to use")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls in seconds")
    
    args = parser.parse_args()
    
    # Load trait definitions
    trait_definitions = load_trait_definitions(args.traits_file)
    
    # Determine which traits to process
    if args.traits:
        traits_to_process = args.traits
        # Validate that all requested traits exist
        for trait in traits_to_process:
            if trait not in trait_definitions:
                print(f"Warning: Trait '{trait}' not found in definitions")
    else:
        traits_to_process = list(trait_definitions.keys())
    
    print(f"Processing traits: {traits_to_process}")
    
    # Initialize generator
    generator = ClaudeDataGenerator(model=args.model)
    
    # Generate data for each trait
    for trait in tqdm(traits_to_process, desc="Generating trait data"):
        try:
            if trait not in trait_definitions:
                print(f"Skipping {trait} - no definition found")
                continue
                
            trait_def = trait_definitions[trait]
            
            # Generate data
            data = generator.generate_trait_data(
                trait=trait,
                trait_instruction=trait_def["instruction"],
                question_instruction=trait_def.get("question_instruction", "")
            )
            
            # Save data
            generator.save_trait_data(trait, data, args.output_dir)
            
            # Add delay between API calls
            if args.delay > 0:
                time.sleep(args.delay)
                
        except Exception as e:
            print(f"Error processing trait '{trait}': {e}")
            continue
    
    print("? Trait data generation complete!")


if __name__ == "__main__":
    main()
