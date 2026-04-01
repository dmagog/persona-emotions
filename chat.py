#!/usr/bin/env python3
"""
Minimal Chat Interface with Persona Vector Steering

A simple chat script that loads a model with optional persona vector steering
and provides an interactive command-line interface.

Usage:
    # Basic chat without steering
    python chat.py --model_path Qwen/Qwen2.5-7B-Instruct

    # Chat with persona steering
    python chat.py --model_path Qwen/Qwen2.5-7B-Instruct --persona inventive --coef 1.0

    # Chat with custom persona vector
    python chat.py --model_path /path/to/model --vector_path custom_vector.pt --coef 1.5 --layer 20
"""

import argparse
import torch
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from eval.model_utils import load_model
from activation_steer import ActivationSteerer

class MinimalChat:
    def __init__(self, model_path: str, vector_path: str = None, coef: float = 1.0, 
                 layer: int = 20, max_tokens: int = 512, temperature: float = 0.7):
        """Initialize chat with model and optional steering"""
        print(f"Loading model: {model_path}")
        self.model, self.tokenizer = load_model(model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.layer = layer
        self.coef = coef
        
        # Load persona vector if provided
        self.vector = None
        if vector_path:
            print(f"Loading persona vector: {vector_path}")
            vector_data = torch.load(vector_path, map_location='cpu')
            self.vector = vector_data[layer]
            print(f"Persona vector loaded with coefficient {coef}")
        
        # Conversation history
        self.history = []
        
    def format_prompt(self, message: str) -> str:
        """Format message as chat prompt"""
        # Add to history
        self.history.append({"role": "user", "content": message})
        
        # Format based on model type
        model_name = self.tokenizer.name_or_path.lower() if hasattr(self.tokenizer, 'name_or_path') else ""
        
        if "qwen" in model_name:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for msg in self.history:
                role = msg["role"]
                content = msg["content"]
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
        elif "llama" in model_name:
            prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
            for msg in self.history:
                role = msg["role"]
                content = msg["content"]
                prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Generic format
            prompt = "System: You are a helpful assistant.\n\n"
            for msg in self.history:
                role = msg["role"].title()
                content = msg["content"]
                prompt += f"{role}: {content}\n\n"
            prompt += "Assistant: "
        
        return prompt
    
    def generate_response(self, message: str) -> str:
        """Generate response with optional steering"""
        prompt = self.format_prompt(message)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with or without steering
        if self.vector is not None:
            vector = self.vector.to(device)
            with ActivationSteerer(self.model, vector, coeff=self.coef, 
                                 layer_idx=self.layer-1, positions="response"):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=self.temperature > 0,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
        else:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Add to history
        self.history.append({"role": "assistant", "content": response.strip()})
        
        return response.strip()
    
    def chat(self):
        """Start interactive chat session"""
        print("\n=== Minimal Chat with Persona Steering ===")
        if self.vector is not None:
            print(f"Steering: ON (coefficient: {self.coef}, layer: {self.layer})")
        else:
            print("Steering: OFF")
        print("Type 'quit' or 'exit' to end the chat.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def find_persona_vector(model_path: str, persona: str) -> str:
    """Find persona vector file for given model and persona"""
    model_name = Path(model_path).name
    vectors_dir = Path("persona_vectors") / model_name
    
    # Try different file patterns
    patterns = [
        f"{persona}_response_avg_diff.pt",
        f"{persona}_prompt_avg_diff.pt", 
        f"{persona}_1.0.pt",
        f"{persona}.pt"
    ]
    
    for pattern in patterns:
        vector_file = vectors_dir / pattern
        if vector_file.exists():
            return str(vector_file)
    
    raise FileNotFoundError(f"No persona vector found for '{persona}' in {vectors_dir}")

def main():
    parser = argparse.ArgumentParser(description="Minimal chat with persona steering")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct", help="Path to model or HuggingFace model ID")
    parser.add_argument("--persona", default=None, type=str, help="Persona name (e.g., 'inventive', 'outgoing')")
    parser.add_argument("--vector_path", default=None, type=str, help="Direct path to persona vector file")
    parser.add_argument("--coef", type=float, default=1.0, help="Steering coefficient")
    parser.add_argument("--layer", type=int, default=20, help="Layer for steering")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Determine vector path
    vector_path = None
    if args.vector_path:
        vector_path = args.vector_path
    elif args.persona:
        try:
            vector_path = find_persona_vector(args.model_path, args.persona)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    # Initialize and start chat
    chat = MinimalChat(
        model_path=args.model_path,
        vector_path=vector_path,
        coef=args.coef,
        layer=args.layer,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    chat.chat()

if __name__ == "__main__":
    main()