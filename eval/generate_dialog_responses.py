#!/usr/bin/env python3
"""
Generate Model Responses for Dialog Benchmark using Local Models

This script takes a dialog benchmark and generates responses using local LLM models.
The responses can then be evaluated using eval_dialog_benchmark.py.

Features:
- Uses model_utils.load_model() and load_vllm_model() for loading models with LoRA support
- Supports both transformers and vLLM backends
- Incremental saving to prevent data loss
- Proper session history handling across multiple turns
- Continuation from existing results

Examples:
    # Basic usage with transformers backend
    python generate_dialog_responses.py \
        --model_path Qwen/Qwen2.5-7B-Instruct \
        --benchmark_file benchmark_evaluation.json \
        --output_file responses.json

    # Using vLLM backend for faster inference
    python generate_dialog_responses.py \
        --model_path meta-llama/Llama-2-7b-chat-hf \
        --benchmark_file benchmark_evaluation.json \
        --output_file responses.json \
        --use_vllm \
        --max_new_tokens 512

    # Continue from existing responses (incremental generation)
    python generate_dialog_responses.py \
        --model_path /path/to/local/model \
        --benchmark_file benchmark_evaluation.json \
        --output_file responses.json \
        --temperature 0.3

Requirements:
    - transformers
    - torch
    - tqdm
    - peft (for LoRA support)
    - vllm (optional, for faster inference)
"""

import json
import argparse
import os
import sys
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
from tqdm import tqdm
import torch
from model_utils import load_model, load_vllm_model
import gc
from collections import defaultdict
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import activation steering from parent directory
from activation_steer import ActivationSteerer

# BFI dimension mapping from eval_bfi.py
BFI_DIMENSION_MAPPING = {
    "Extraversion": "outgoing",
    "Agreeableness": "compassionate", 
    "Conscientiousness": "dependable",
    "Neuroticism": "nervous",
    "Openness": "inventive"
}

def predict_bfi_traits(model, tokenizer, persona_context: str, current_input: str) -> Dict[str, float]:
    """
    Predict BFI trait adjustments for the current context and persona.
    Returns incremental coefficients for steering (not absolute trait levels).
    The base model already has: high agreeableness & conscientiousness, 
    moderate extraversion & openness, low neuroticism.
    """
    prediction_prompt = f"""You are a Personality Analyst AI. Your task is to determine the most appropriate **situational adjustments (deltas)** to a baseline AI personality based on the specific context and interaction needs.

**Baseline Personality Profile:**
The AI has these default traits:
- **Agreeableness:** High (cooperative, trusting, helpful)
- **Conscientiousness:** High (organized, reliable, disciplined)
- **Extraversion:** Moderate (balanced social energy)
- **Openness:** Moderate (balanced creativity and practicality)
- **Neuroticism:** Low (generally calm and stable)

**Context to Analyze:**
- **Persona Context:** {persona_context}
- **Current Input:** {current_input}

**Your Task:**
Determine which traits need adjustment (-2.0 to +2.0) based on what would be most effective for this specific interaction. Consider both directions equally and choose based on situational demands.

**Trait Adjustment Guidelines:**

**Extraversion:**
- **Increase (+)** for: Group activities, public speaking, networking events, team leadership, social gatherings, collaborative brainstorming
- **Decrease (-)** for: Individual focused work, quiet reflection, private study, solo creative tasks, introspective discussions

**Agreeableness:**
- **Increase (+)** for: Conflict resolution, team building, emotional support, consensus building, customer service
- **Decrease (-)** for: Critical feedback, boundary setting, competitive situations, principled stands, resource constraints

**Conscientiousness:**
- **Increase (+)** for: Detailed planning, precision work, deadline management, quality control, formal procedures
- **Decrease (-)** for: Spontaneous responses, creative brainstorming, flexible adaptation, crisis situations requiring speed

**Neuroticism:**
- **Increase (+)** for: Appropriate caution, emotional sensitivity, vulnerability, risk awareness, vigilant situations
- **Decrease (-)** for: Calm leadership, confident decisions, reassuring others, crisis management, stability needs

**Openness:**
- **Increase (+)** for: Creative problem-solving, exploring new ideas, intellectual curiosity, innovation, novel situations
- **Decrease (-)** for: Following procedures, traditional approaches, proven solutions, consistency needs, established protocols

**Decision Principles:**
1. **Situational Fit:** Choose traits that best serve the interaction goals
2. **Context Sensitivity:** Consider what the human needs from this specific interaction
3. **Balanced Assessment:** Evaluate both positive and negative adjustments equally
4. **Natural Baseline:** Use 0.0 when baseline personality already fits the situation well

**Output Format:**
Provide only the numerical adjustment scores:

Extraversion: [score]
Agreeableness: [score]
Conscientiousness: [score]
Neuroticism: [score]
Openness: [score]"""

    try:
        # Format as conversation
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prediction_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted_prompt = f"<|im_start|>user\n{prediction_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,  # Lower temperature for more consistent predictions
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # Parse trait scores from response
        trait_scores = {}
        for trait in BFI_DIMENSION_MAPPING.keys():
            pattern = rf"{trait}:\s*([-]?\d+\.?\d*)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Clamp to valid range
                score = max(-2.0, min(2.0, score))
                trait_scores[trait] = score
            else:
                trait_scores[trait] = 0.0  # Default to neutral
        
        return trait_scores
        
    except Exception as e:
        print(f"Warning: BFI trait prediction failed: {e}")
        # Return neutral scores as fallback
        return {trait: 0.0 for trait in BFI_DIMENSION_MAPPING.keys()}

def load_persona_vectors(model_path: str, persona_vectors_dir: str = None, steering_layer: int = 20) -> Dict[str, torch.Tensor]:
    """Load persona vectors for the given model."""
    if persona_vectors_dir is None:
        # Default to persona_vectors directory in project root
        project_root = Path(__file__).parent.parent
        persona_vectors_dir = project_root / "persona_vectors"
    
    model_name = Path(model_path).name
    vectors_path = Path(persona_vectors_dir) / model_name
    
    if not vectors_path.exists():
        print(f"Warning: Persona vectors not found for model {model_name} at {vectors_path}")
        return {}
    
    vectors = {}
    try:
        for trait_name in BFI_DIMENSION_MAPPING.values():
            # Try different vector file patterns
            for pattern in [f"{trait_name}_response_avg_diff.pt"]:
                vector_file = vectors_path / pattern
                if vector_file.exists():
                    vectors[trait_name] = torch.load(vector_file, map_location='cpu')[steering_layer]
                    break
            
            if trait_name not in vectors:
                print(f"Warning: Vector not found for trait {trait_name}")
        
        print(f"Loaded {len(vectors)} persona vectors from {vectors_path}")
        return vectors
        
    except Exception as e:
        print(f"Error loading persona vectors: {e}")
        return {}

def apply_steering_to_response(model, tokenizer, conversation_history: List[Dict[str, str]], 
                             trait_scores: Dict[str, float], persona_vectors: Dict[str, torch.Tensor],
                             layer: int = 16, steering_type: str = "response", 
                             max_tokens: int = 512, temperature: float = 0.7,
                             system_prompt: str = "") -> str:
    """
    Apply persona vector steering based on predicted BFI traits and generate response.
    This is adapted from sample_steering function in eval_bfi.py.
    """
    try:
        
        # Apply steering for each trait with non-zero coefficient
        device = next(model.parameters()).device
        
        # Combine multiple trait vectors if needed
        active_vectors = []
        active_coeffs = []
        
        for trait, coeff in trait_scores.items():
            # Only apply adjustments that are significant (>0.2) to avoid over-steering
            # This respects the model's baseline personality and only makes necessary adjustments
            if abs(coeff) > 0.2 and trait in BFI_DIMENSION_MAPPING:  
                trait_vector_name = BFI_DIMENSION_MAPPING[trait]
                if trait_vector_name in persona_vectors:
                    vector = persona_vectors[trait_vector_name].to(device)
                    # Scale down the coefficient to be more conservative (0.5x)
                    # This ensures we enhance/suppress rather than override baseline traits
                    scaled_coeff = coeff * 0.5
                    active_vectors.append(vector * scaled_coeff)
                    active_coeffs.append(scaled_coeff)
                    print(f"Applying {trait} adjustment: {coeff:.2f} -> {scaled_coeff:.2f}")
        
        if not active_vectors:
            # No significant adjustments needed, use baseline personality (normal generation)
            print("No significant trait adjustments needed, using baseline personality")
            prompt = format_chat_prompt(tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else "", 
                                      system_prompt, conversation_history)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()
        
        # Combine vectors (simple average for multiple traits)
        combined_vector = torch.stack(active_vectors).sum(dim=0)
        combined_coeff = 1.0  # Average magnitude
        
        # Apply steering using ActivationSteerer
        with ActivationSteerer(model, combined_vector, coeff=combined_coeff, 
                              layer_idx=layer-1, positions=steering_type):
            
            # Format prompt with system prompt included
            prompt = format_chat_prompt(tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else "", 
                                      system_prompt, conversation_history)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with steering
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()
    
    except Exception as e:
        print(f"Error in steering response generation: {e}")
        # Fallback to normal generation
        return "[Error in steered generation]"

def load_benchmark_evaluation_format(benchmark_file: str) -> List[Dict[str, Any]]:
    """Load benchmark in evaluation format"""
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_existing_responses(output_file: str) -> tuple[Dict[str, Any], set]:
    """Load existing responses and return metadata and completed items"""
    if not os.path.exists(output_file):
        return {}, set()
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        completed_items = set()
        if "responses" in data:
            for response in data["responses"]:
                key = f"{response['session_id']}_{response['turn_number']}"
                completed_items.add(key)
        
        return data.get("metadata", {}), completed_items
    except Exception as e:
        print(f"Warning: Could not load existing responses: {e}")
        return {}, set()

def save_responses_incrementally(responses: List[Dict], metadata: Dict, output_file: str):
    """Save responses incrementally"""
    output_data = {
        "metadata": metadata,
        "responses": responses
    }
    
    # Write to temporary file first, then rename for atomic operation
    temp_file = output_file + ".tmp"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    os.rename(temp_file, output_file)

def format_chat_prompt(model_path: str, system_prompt: str, history: List[Dict[str, str]]) -> str:
    """Format chat prompt based on model type"""
    model_name_lower = model_path.lower()
    
    if "llama" in model_name_lower:
        # Llama format
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
    elif "qwen" in model_name_lower:
        # Qwen format
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
    elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
        # Mistral format - combine system and first user message
        if history and history[0]["role"] == "user":
            first_user = history[0]["content"]
            combined_prompt = f"{system_prompt}\n\n{first_user}"
            prompt = f"<s>[INST] {combined_prompt} [/INST]"
            
            # Add remaining history
            for msg in history[1:]:
                if msg["role"] == "user":
                    prompt += f" [INST] {msg['content']} [/INST]"
                else:  # assistant
                    prompt += f" {msg['content']}</s>"
        else:
            prompt = f"<s>[INST] {system_prompt} [/INST]"
            
    else:
        # Generic format
        prompt = f"System: {system_prompt}\n\n"
        for msg in history:
            role = msg["role"].title()
            content = msg["content"]
            prompt += f"{role}: {content}\n\n"
        prompt += "Assistant:"
    
    return prompt

class TransformersModelGenerator:
    """Model generator using transformers backend"""
    
    def __init__(self, model_path: str, dtype=torch.bfloat16, max_new_tokens: int = 512, 
                 dynamic_steering: bool = False, persona_vectors_dir: str = None,
                 steering_layer: int = 16, steering_type: str = "response"):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.dynamic_steering = dynamic_steering
        self.steering_layer = steering_layer
        self.steering_type = steering_type
        
        print(f"Loading model with transformers: {model_path}")
        self.model, self.tokenizer = load_model(model_path, dtype=dtype)
        print(f"Model loaded successfully")
        
        # Load persona vectors if dynamic steering is enabled
        self.persona_vectors = {}
        if dynamic_steering:
            print("Loading persona vectors for dynamic steering...")
            self.persona_vectors = load_persona_vectors(model_path, persona_vectors_dir, steering_layer)
            if not self.persona_vectors:
                print("Warning: Dynamic steering enabled but no persona vectors loaded. Falling back to normal generation.")
                self.dynamic_steering = False
    
    def generate_response(self, system_prompt: str, history: List[Dict[str, str]], 
                         temperature: float = 0.7, persona_context: str = None,
                         current_input: str = None) -> tuple[str, Dict[str, float]]:
        """Generate response with conversation history and optional dynamic steering
        Returns: (response_text, trait_scores)
        """
        trait_scores = None  # Initialize trait_scores
        
        try:
            # If dynamic steering is enabled and we have the necessary context
            if (self.dynamic_steering and self.persona_vectors and 
                persona_context and current_input):
                
                print("Predicting BFI trait adjustments for dynamic steering...")
                trait_scores = predict_bfi_traits(self.model, self.tokenizer, 
                                                persona_context, current_input)
                
                print(f"Predicted trait adjustments: {trait_scores}")
                
                # Apply steering if any traits have significant adjustment scores
                significant_adjustments = {k: v for k, v in trait_scores.items() if abs(v) > 0.2}
                if significant_adjustments:
                    print(f"Applying incremental steering for: {significant_adjustments}")
                    response = apply_steering_to_response(
                        self.model, self.tokenizer, history, trait_scores, 
                        self.persona_vectors, self.steering_layer, self.steering_type,
                        self.max_new_tokens, temperature, system_prompt
                    )
                    return response, trait_scores
                else:
                    print("No significant trait adjustments needed, using baseline personality")
            
            # Normal generation without steering
            prompt = format_chat_prompt(self.model_path, system_prompt, history)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            
            # Move to model's device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip(), trait_scores
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[Error: {str(e)}]", trait_scores
    
    def cleanup(self):
        """Clean up model and free memory"""
        del self.model
        del self.tokenizer
        if hasattr(self, 'persona_vectors'):
            del self.persona_vectors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class VLLMModelGenerator:
    """Model generator using vLLM backend"""
    
    def __init__(self, model_path: str, max_new_tokens: int = 512, 
                 dynamic_steering: bool = False, persona_vectors_dir: str = None,
                 steering_layer: int = 16, steering_type: str = "response"):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.dynamic_steering = dynamic_steering
        self.steering_layer = steering_layer
        self.steering_type = steering_type
        
        print(f"Loading model with vLLM: {model_path}")
        self.llm, self.tokenizer, self.lora_path = load_vllm_model(model_path)
        print(f"Model loaded successfully")
        
        # For vLLM, we need to note that dynamic steering is limited
        # vLLM doesn't expose model internals for activation steering
        self.persona_vectors = {}
        if dynamic_steering:
            print("Warning: Dynamic steering with vLLM is not fully supported.")
            print("vLLM doesn't expose model layers for activation steering.")
            print("Falling back to normal generation.")
            self.dynamic_steering = False
    
    def generate_response(self, system_prompt: str, history: List[Dict[str, str]], 
                         temperature: float = 0.7, persona_context: str = None,
                         current_input: str = None) -> tuple[str, Dict[str, float]]:
        """Generate response with conversation history using vLLM
        Returns: (response_text, trait_scores) - trait_scores will be None for vLLM
        """
        try:
            from vllm import SamplingParams
            
            # Note: Dynamic steering is not supported with vLLM
            if self.dynamic_steering:
                print("Note: Dynamic steering requested but not supported with vLLM backend")
            
            prompt = format_chat_prompt(self.model_path, system_prompt, history)
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                repetition_penalty=1.1
            )
            
            # Generate with LoRA if available
            if self.lora_path:
                outputs = self.llm.generate(
                    prompt, 
                    sampling_params,
                    lora_request={"lora_name": "default", "lora_path": self.lora_path}
                )
            else:
                outputs = self.llm.generate(prompt, sampling_params)
            
            return outputs[0].outputs[0].text.strip(), None  # No trait scores for vLLM
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"[Error: {str(e)}]", None
    
    def cleanup(self):
        """Clean up model and free memory"""
        # vLLM handles cleanup automatically
        pass

def generate_responses_for_benchmark(model_path: str, benchmark_file: str, 
                                   output_file: str, temperature: float = 0.7,
                                   dtype=torch.bfloat16, max_new_tokens: int = 512,
                                   use_vllm: bool = False, dynamic_steering: bool = False,
                                   persona_vectors_dir: str = None, steering_layer: int = 16,
                                   steering_type: str = "response"):
    """Generate responses for entire benchmark using local model with session history"""
    
    print("Loading benchmark data...")
    benchmark_data = load_benchmark_evaluation_format(benchmark_file)
    
    # Load existing responses
    existing_metadata, completed_items = load_existing_responses(output_file)
    print(f"Found {len(completed_items)} existing responses")
    
    # Group benchmark data by session
    sessions = defaultdict(list)
    for item in benchmark_data:
        sessions[item["session_id"]].append(item)
    
    # Sort turns within each session
    for session_id in sessions:
        sessions[session_id].sort(key=lambda x: x["turn_number"])
    
    print(f"Processing {len(sessions)} sessions with {len(benchmark_data)} total turns")
    
    # Initialize model generator
    if use_vllm:
        generator = VLLMModelGenerator(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            dynamic_steering=dynamic_steering,
            persona_vectors_dir=persona_vectors_dir,
            steering_layer=steering_layer,
            steering_type=steering_type
        )
    else:
        generator = TransformersModelGenerator(
            model_path=model_path,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            dynamic_steering=dynamic_steering,
            persona_vectors_dir=persona_vectors_dir,
            steering_layer=steering_layer,
            steering_type=steering_type
        )
    
    # Prepare metadata
    metadata = {
        "generation_time": time.time(),
        "model_path": model_path,
        "temperature": temperature,
        "dtype": str(dtype) if not use_vllm else "vllm_managed",
        "max_new_tokens": max_new_tokens,
        "use_vllm": use_vllm,
        "dynamic_steering": dynamic_steering,
        "steering_layer": steering_layer if dynamic_steering else None,
        "steering_type": steering_type if dynamic_steering else None,
        "persona_vectors_dir": persona_vectors_dir if dynamic_steering else None,
        "stores_trait_scores": dynamic_steering and not use_vllm,  # Only transformers backend stores trait scores
        "trait_score_format": "BFI dimensions: Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness" if dynamic_steering and not use_vllm else None,
        "total_sessions": len(sessions),
        "total_turns": len(benchmark_data)
    }
    
    # Update with existing metadata if available
    if existing_metadata:
        metadata.update(existing_metadata)
        metadata["continued_generation"] = True
        metadata["continuation_time"] = time.time()
    
    all_responses = []
    
    # Load existing responses
    if completed_items:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                all_responses = existing_data.get("responses", [])
        except Exception as e:
            print(f"Warning: Could not load existing responses: {e}")
    
    try:
        # Process sessions with progress tracking
        total_new_items = 0
        session_progress = tqdm(sessions.items(), desc="Processing sessions")
        
        for session_id, session_turns in session_progress:
            # Track conversation history for this session
            conversation_history = []
            system_prompt = session_turns[0]["system_prompt"] + " Stay in first-person voice, expressing thoughts, feelings, and reactions as a real person would."  # Same for all turns in session

            session_progress.set_description(f"Session: {session_id}")
            
            for turn in session_turns:
                item_key = f"{session_id}_{turn['turn_number']}"
                
                # Skip if already completed
                if item_key in completed_items:
                    # Load existing response into history for context
                    existing_response = next(
                        (r for r in all_responses 
                         if r["session_id"] == session_id and r["turn_number"] == turn["turn_number"]), 
                        None
                    )
                    if existing_response:
                        conversation_history.append({"role": "user", "content": turn["model_input"]})
                        conversation_history.append({"role": "assistant", "content": existing_response["model_response"]})
                    continue
                
                # Add current user input to history
                conversation_history.append({"role": "user", "content": turn["model_input"]})
                
                # Prepare context for dynamic steering if enabled
                persona_context = None
                current_input = None
                if dynamic_steering:
                    # Build persona context from turn data
                    persona_context = f"""Name: {turn.get('persona_name', 'Unknown')}
Role: {turn.get('persona_role', 'Unknown')}
System Prompt: {system_prompt}
Background: {turn.get('background', '')}"""
                    
                    current_input = turn["model_input"]
                
                # Generate response with full conversation history and steering context
                response_text, trait_scores = generator.generate_response(
                    system_prompt, conversation_history, temperature,
                    persona_context=persona_context, current_input=current_input
                )
                
                # Add assistant response to history for next turn
                conversation_history.append({"role": "assistant", "content": response_text})
                
                # Create response item
                response_item = {
                    "session_id": session_id,
                    "turn_number": turn["turn_number"],
                    "persona_name": turn["persona_name"],
                    "persona_role": turn["persona_role"],
                    "model_input": turn["model_input"],
                    "model_response": response_text,
                    "expected_emotion": turn["expected_emotion"],
                    "expected_response_style": turn.get("expected_response_style", ""),
                    "scenario_description": turn.get("scenario_description", ""),
                    "context": turn["context"],
                    "conversation_history": conversation_history.copy(),  # Store full history
                    "predicted_trait_scores": trait_scores  # Store trait scores if available
                }
                
                all_responses.append(response_item)
                completed_items.add(item_key)
                total_new_items += 1
                
                # Update metadata
                metadata["num_responses"] = len(all_responses)
                metadata["last_updated"] = time.time()
                
                # Incremental save every 5 responses
                if total_new_items % 5 == 0:
                    save_responses_incrementally(all_responses, metadata, output_file)
                    session_progress.set_postfix({"saved": len(all_responses)})
    
    finally:
        # Always cleanup the model and save final results
        print("Cleaning up model...")
        generator.cleanup()
        
        # Final save
        save_responses_incrementally(all_responses, metadata, output_file)
    
    print(f"Generated {total_new_items} new responses")
    print(f"Total responses: {len(all_responses)}")
    print(f"Responses saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate model responses for dialog benchmark using local models")
    parser.add_argument("--model_path", required=True, help="Path to local model (HuggingFace model name or local path)")
    parser.add_argument("--benchmark_file", required=True, help="Path to benchmark evaluation file")
    parser.add_argument("--output_file", required=True, help="Path to save model responses")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--dtype", default="bfloat16", help="Data type (bfloat16, float16, float32) - ignored for vLLM")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for faster inference")
    
    # Dynamic steering options
    parser.add_argument("--dynamic_steering", action="store_true", 
                       help="Enable dynamic BFI trait-based persona steering")
    parser.add_argument("--persona_vectors_dir", type=str, default=None,
                       help="Path to persona vectors directory (default: ./persona_vectors)")
    parser.add_argument("--steering_layer", type=int, default=20,
                       help="Layer index for activation steering (default: 20)")
    parser.add_argument("--steering_type", type=str, default="response", choices=["response", "last", "prompt"],
                       help="Type of steering application (default: response)")
    
    args = parser.parse_args()
    
    # Convert dtype string to torch dtype (only used for transformers backend)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    
    if args.dtype not in dtype_map:
        print(f"Error: Invalid dtype '{args.dtype}'. Choose from: {list(dtype_map.keys())}")
        return
    
    dtype = dtype_map[args.dtype]
    
    # Check dynamic steering compatibility
    if args.dynamic_steering and args.use_vllm:
        print("Warning: Dynamic steering is not supported with vLLM backend. Continuing without steering.")
        args.dynamic_steering = False
    
    # Check vLLM availability
    if args.use_vllm:
        try:
            import vllm
            print("Using vLLM backend for generation")
        except ImportError:
            print("Error: vLLM not installed. Install with: pip install vllm")
            print("Falling back to transformers backend...")
            args.use_vllm = False
    
    if args.dynamic_steering:
        print("Dynamic BFI trait-based steering enabled")
        print(f"- Persona vectors directory: {args.persona_vectors_dir or 'default (./persona_vectors)'}")
        print(f"- Steering layer: {args.steering_layer}")
        print(f"- Steering type: {args.steering_type}")
        
        if not args.use_vllm:
            # Check if activation steering is available
            try:
                from activation_steer import ActivationSteerer
                print("ActivationSteerer imported successfully")
            except ImportError:
                print("Error: activation_steer module not found. Dynamic steering requires ActivationSteerer.")
                return
    
    generate_responses_for_benchmark(
        args.model_path,
        args.benchmark_file,
        args.output_file,
        args.temperature,
        dtype,
        args.max_new_tokens,
        args.use_vllm,
        args.dynamic_steering,
        args.persona_vectors_dir,
        args.steering_layer,
        args.steering_type
    )

if __name__ == "__main__":
    main()
