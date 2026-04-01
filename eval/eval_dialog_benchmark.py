#!/usr/bin/env python3
"""
Multi-Session Dialog Benchmark Evaluator

This script evaluates LLM responses on the multi-session dialog benchmark using LLM judges.
It measures four key dimensions:
1. Trait Adherence: How well the response reflects the expected emotion
2. Role Consistency: How well the agent maintains its core persona
3. Rubric Conformance: How well the response follows the expected response style
4. Engagingness: How natural and human-like the conversation is
"""

import json
import argparse
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import openai
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

@dataclass
class EvaluationResult:
    """Results from evaluating a single turn"""
    session_id: str
    turn_number: int
    persona_name: str
    persona_role: str
    trait_adherence: float
    role_consistency: float
    rubric_conformance: float
    engagingness: float
    insightfulness: float
    judge_reasoning: str
    model_response: str
    expected_emotion: str
    expected_response_style: str

@dataclass
class SessionEvaluation:
    """Aggregated evaluation results for a complete session"""
    session_id: str
    persona_name: str
    persona_role: str
    num_turns: int
    avg_trait_adherence: float
    avg_role_consistency: float
    avg_rubric_conformance: float
    avg_engagingness: float
    avg_insightfulness: float
    overall_score: float
    turn_results: List[EvaluationResult]

class DialogBenchmarkEvaluator:
    """Evaluates LLM responses on the dialog benchmark using LLM judges"""
    
    def __init__(self, judge_api_key: str, judge_model: str = "gpt-4", temperature: float = 0.1, max_retries: int = 3):
        self.judge_client = openai.OpenAI(api_key=judge_api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_retries = max_retries
        
    def evaluate_turn(self, benchmark_turn: Dict[str, Any], model_response: str) -> EvaluationResult:
        """Evaluate a single turn using LLM judge with retry logic"""
        
        # Extract information from benchmark
        session_id = benchmark_turn["session_id"]
        turn_number = benchmark_turn["turn_number"]
        persona_name = benchmark_turn["persona_name"]
        persona_role = benchmark_turn["persona_role"]
        system_prompt = benchmark_turn["system_prompt"]
        model_input = benchmark_turn["model_input"]
        context = benchmark_turn["context"]
        expected_emotion = benchmark_turn["expected_emotion"]
        expected_response_style = benchmark_turn["expected_response_style"]
        
        # Create evaluation prompt
        judge_prompt = f"""You are an expert evaluator of conversational AI systems. Please evaluate the following LLM response across four dimensions on a 1-5 scale (1=Poor, 2=Below Average, 3=Average, 4=Good, 5=Excellent).

**CONTEXT:**
Persona: {persona_name} - {persona_role}
System Prompt: {system_prompt}

**SCENARIO:**]
Model Input: "{model_input}"
Context: {context}

**EXPECTED BEHAVIOR:**
Expected Emotion: {expected_emotion}
Expected Response Style: {expected_response_style}

**MODEL RESPONSE TO EVALUATE:**
"{model_response}"

**EVALUATION CRITERIA:**

1. **Trait Adherence (1-5):** How well does the response reflect the expected emotion "{expected_emotion}"?
   - 5: Perfectly captures the emotion with appropriate intensity and expression
   - 4: Clearly shows the emotion with good authenticity
   - 3: Shows some evidence of the emotion but could be stronger
   - 2: Minimal evidence of the expected emotion
   - 1: No evidence of the expected emotion or contradicts it

2. **Role Consistency (1-5):** How well does the response maintain the core persona without breaking character?
   - 5: Perfectly consistent with persona, uses appropriate language/behavior
   - 4: Mostly consistent, minor deviations that don't break character
   - 3: Generally consistent but some noticeable inconsistencies
   - 2: Several inconsistencies that somewhat break character
   - 1: Major inconsistencies or completely out of character

3. **Rubric Conformance (1-5):** How well does the response match the expected response style?
   - 5: Perfectly follows the expected style guidelines and nuances
   - 4: Closely follows most style guidelines with minor deviations
   - 3: Follows general style but misses some important nuances
   - 2: Partially follows style but significant gaps
   - 1: Poorly follows or ignores the expected response style

4. **Engagingness (1-5):** How natural, human-like, and compelling is the response?
   - 5: Highly engaging, feels completely natural and human-like
   - 4: Engaging and natural with minor artificial elements
   - 3: Moderately engaging, some natural elements
   - 2: Somewhat engaging but noticeably artificial
   - 1: Not engaging, feels robotic or unnatural

5. **Insightfulness (1-5):** How much depth, thoughtfulness, and analytical reasoning does the response demonstrate?
   - 5: Highly insightful, providing deep analysis of the situation, emotions, and potential actions with sophisticated understanding
   - 4: Shows very good insight and reasoning, moving well beyond superficial responses with clear analytical depth
   - 3: Contains good thoughtful elements with some analytical reasoning, though not consistently deep
   - 2: Shows moderate insight with some thoughtful observations, but limited analytical depth
   - 1: Lacks significant insight; the response is mostly superficial or purely narrative

**RESPONSE FORMAT:**
Provide your evaluation in JSON format:
{{
  "trait_adherence": <score>,
  "role_consistency": <score>,
  "rubric_conformance": <score>,
  "engagingness": <score>,
  "insightfulness": <score>,
  "reasoning": "Detailed explanation of your scoring rationale, analyzing each dimension and providing specific examples from the response."
}}"""

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
                
                response_content = response.choices[0].message.content
                
                # Clean up and parse JSON response
                if response_content.strip().startswith('```'):
                    lines = response_content.strip().split('\n')
                    start_idx = 0
                    end_idx = len(lines)
                    
                    for i, line in enumerate(lines):
                        if line.strip().startswith('```'):
                            if start_idx == 0:
                                start_idx = i + 1
                            else:
                                end_idx = i
                                break
                    
                    response_content = '\n'.join(lines[start_idx:end_idx])
                
                judge_result = json.loads(response_content)
                
                # Validate that all required fields are present and valid
                required_fields = ["trait_adherence", "role_consistency", "rubric_conformance", "engagingness", "insightfulness", "reasoning"]
                for field in required_fields:
                    if field not in judge_result:
                        raise ValueError(f"Missing required field: {field}")
                    if field != "reasoning" and not isinstance(judge_result[field], (int, float)):
                        raise ValueError(f"Invalid score type for {field}: {type(judge_result[field])}")
                    if field != "reasoning" and not (1 <= judge_result[field] <= 5):
                        raise ValueError(f"Score out of range for {field}: {judge_result[field]}")
                
                # Successfully parsed and validated
                return EvaluationResult(
                    session_id=session_id,
                    turn_number=turn_number,
                    persona_name=persona_name,
                    persona_role=persona_role,
                    trait_adherence=float(judge_result["trait_adherence"]),
                    role_consistency=float(judge_result["role_consistency"]),
                    rubric_conformance=float(judge_result["rubric_conformance"]),
                    engagingness=float(judge_result["engagingness"]),
                    insightfulness=float(judge_result["insightfulness"]),
                    judge_reasoning=judge_result["reasoning"],
                    model_response=model_response,
                    expected_emotion=expected_emotion,
                    expected_response_style=expected_response_style
                )
                
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1}/{self.max_retries} failed for turn {turn_number} in session {session_id}: {e}")
                
                # Add exponential backoff delay
                if attempt < self.max_retries - 1:
                    delay = 2 ** attempt  # 1s, 2s, 4s, etc.
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        # All retries failed - raise the error to stop evaluation
        error_msg = f"Failed to evaluate turn {turn_number} for {persona_name} after {self.max_retries} attempts. Last error: {last_error}"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)
    
    def evaluate_session(self, benchmark_session: List[Dict[str, Any]], 
                        model_responses: List[str]) -> SessionEvaluation:
        """Evaluate all turns in a session"""
        
        if len(benchmark_session) != len(model_responses):
            raise ValueError(f"Mismatch: {len(benchmark_session)} benchmark turns vs {len(model_responses)} responses")
        
        turn_results = []
        session_id = benchmark_session[0]["session_id"]
        persona_name = benchmark_session[0]["persona_name"]
        persona_role = benchmark_session[0]["persona_role"]
        
        print(f"Evaluating session: {persona_name} ({len(benchmark_session)} turns)")
        
        for turn_data, response in zip(benchmark_session, model_responses):
            try:
                result = self.evaluate_turn(turn_data, response)
                turn_results.append(result)
                
                # Add small delay to avoid rate limits
                time.sleep(0.5)
            except RuntimeError as e:
                # Re-raise the error to stop evaluation of this session and subsequent sessions
                print(f"Critical error in session {session_id}: {e}")
                raise e
        
        # Calculate averages
        avg_trait_adherence = np.mean([r.trait_adherence for r in turn_results])
        avg_role_consistency = np.mean([r.role_consistency for r in turn_results])
        avg_rubric_conformance = np.mean([r.rubric_conformance for r in turn_results])
        avg_engagingness = np.mean([r.engagingness for r in turn_results])
        avg_insightfulness = np.mean([r.insightfulness for r in turn_results])
        
        # Overall score is the average of all dimensions
        overall_score = np.mean([avg_trait_adherence, avg_role_consistency, 
                               avg_rubric_conformance, avg_engagingness, avg_insightfulness])
        
        return SessionEvaluation(
            session_id=session_id,
            persona_name=persona_name,
            persona_role=persona_role,
            num_turns=len(turn_results),
            avg_trait_adherence=avg_trait_adherence,
            avg_role_consistency=avg_role_consistency,
            avg_rubric_conformance=avg_rubric_conformance,
            avg_engagingness=avg_engagingness,
            avg_insightfulness=avg_insightfulness,
            overall_score=overall_score,
            turn_results=turn_results
        )
    
    def load_benchmark_evaluation_format(self, benchmark_file: str) -> List[Dict[str, Any]]:
        """Load benchmark in evaluation format"""
        with open(benchmark_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_model_responses(self, responses_file: str) -> Dict[str, List[str]]:
        """Load model responses organized by session_id"""
        with open(responses_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Debug: Loaded data type: {type(data)}")
        if isinstance(data, dict):
            print(f"Debug: Dict keys: {list(data.keys())}")
        elif isinstance(data, list):
            print(f"Debug: List length: {len(data)}")
            if data:
                print(f"Debug: First item type: {type(data[0])}")
                if isinstance(data[0], dict):
                    print(f"Debug: First item keys: {list(data[0].keys())}")
        
        # Handle different response file formats
        if isinstance(data, dict) and "responses" in data:
            # New format with metadata and responses
            responses_list = data["responses"]
            print(f"Debug: Using 'responses' key, found {len(responses_list)} items")
        elif isinstance(data, list):
            # Old format - direct list of responses
            responses_list = data
            print(f"Debug: Using direct list format with {len(responses_list)} items")
        else:
            raise ValueError(f"Unexpected response file format. Expected dict with 'responses' key or list, got {type(data)}")
        
        if not responses_list:
            raise ValueError("No responses found in file")
        
        # Group responses by session_id and sort by turn_number
        responses_by_session = {}
        for i, item in enumerate(responses_list):
            if not isinstance(item, dict):
                raise ValueError(f"Expected response item {i} to be dict, got {type(item)}: {item}")
            
            # Check for required fields
            if "session_id" not in item:
                print(f"Warning: Response item {i} missing 'session_id': {item.keys()}")
                continue
            if "model_response" not in item:
                print(f"Warning: Response item {i} missing 'model_response': {item.keys()}")
                continue
                
            session_id = item["session_id"]
            turn_number = item.get("turn_number", 0)  # Default to 0 if not present
            
            if session_id not in responses_by_session:
                responses_by_session[session_id] = []
            
            # Store as tuple (turn_number, response) for sorting
            responses_by_session[session_id].append((turn_number, item["model_response"]))
        
        # Sort responses by turn number and extract just the response text
        for session_id in responses_by_session:
            responses_by_session[session_id].sort(key=lambda x: x[0])  # Sort by turn_number
            responses_by_session[session_id] = [response for _, response in responses_by_session[session_id]]
        
        return responses_by_session
    
    def load_existing_evaluations(self, output_file: str) -> tuple[Dict[str, Any], set]:
        """Load existing evaluation results and return metadata and completed sessions"""
        if not os.path.exists(output_file):
            return {}, set()
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            completed_sessions = set()
            if "session_evaluations" in data:
                for session_eval in data["session_evaluations"]:
                    completed_sessions.add(session_eval["session_id"])
            
            return data.get("metadata", {}), completed_sessions
        except Exception as e:
            print(f"Warning: Could not load existing evaluations: {e}")
            return {}, set()
    
    def save_evaluations_incrementally(self, evaluations: List[SessionEvaluation], 
                                     metadata: Dict[str, Any], output_file: str):
        """Save evaluation results incrementally"""
        results_data = {
            "metadata": metadata,
            "summary": self.calculate_summary_stats(evaluations),
            "session_evaluations": [asdict(eval) for eval in evaluations]
        }
        
        # Write to temporary file first, then rename for atomic operation
        temp_file = output_file + ".tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        os.rename(temp_file, output_file)
    
    def evaluate_benchmark(self, benchmark_file: str, responses_file: str, 
                          output_file: str, save_frequency: int = 5) -> List[SessionEvaluation]:
        """Evaluate complete benchmark with incremental saving and continuation support"""
        
        print("Loading benchmark data...")
        benchmark_data = self.load_benchmark_evaluation_format(benchmark_file)
        
        print("Loading model responses...")
        model_responses = self.load_model_responses(responses_file)
        
        # Load existing evaluations
        existing_metadata, completed_sessions = self.load_existing_evaluations(output_file)
        print(f"Found {len(completed_sessions)} existing session evaluations")
        
        # Group benchmark data by session
        sessions_data = {}
        for turn_data in benchmark_data:
            session_id = turn_data["session_id"]
            if session_id not in sessions_data:
                sessions_data[session_id] = []
            sessions_data[session_id].append(turn_data)
        
        # Sort turns within each session
        for session_id in sessions_data:
            sessions_data[session_id].sort(key=lambda x: x["turn_number"])
        
        # Prepare metadata
        metadata = {
            "evaluation_time": time.time(),
            "judge_model": self.judge_model,
            "total_sessions": len(sessions_data),
            "benchmark_file": benchmark_file,
            "responses_file": responses_file,
            "save_frequency": save_frequency
        }
        
        # Update with existing metadata if available
        if existing_metadata:
            metadata.update(existing_metadata)
            metadata["continued_evaluation"] = True
            metadata["continuation_time"] = time.time()
        
        # Load existing evaluations
        session_evaluations = []
        if completed_sessions:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    for session_dict in existing_data.get("session_evaluations", []):
                        # Reconstruct SessionEvaluation objects
                        session_eval = SessionEvaluation(**session_dict)
                        session_evaluations.append(session_eval)
            except Exception as e:
                print(f"Warning: Could not load existing session evaluations: {e}")
        
        print(f"Evaluating {len(sessions_data) - len(completed_sessions)} new sessions...")
        
        new_evaluations = 0
        for session_id, session_turns in tqdm(sessions_data.items(), desc="Evaluating sessions"):
            # Skip if already completed
            if session_id in completed_sessions:
                continue
                
            if session_id in model_responses:
                try:
                    session_eval = self.evaluate_session(session_turns, model_responses[session_id])
                    session_evaluations.append(session_eval)
                    completed_sessions.add(session_id)
                    new_evaluations += 1
                    
                    # Update metadata
                    metadata["num_sessions"] = len(session_evaluations)
                    metadata["last_updated"] = time.time()
                    
                    # Incremental save at specified frequency
                    if new_evaluations % save_frequency == 0:
                        self.save_evaluations_incrementally(session_evaluations, metadata, output_file)
                        print(f"Saved progress: {len(session_evaluations)} sessions evaluated")
                        
                except RuntimeError as e:
                    # Critical error occurred after all retries - stop evaluation
                    print(f"\nCRITICAL ERROR: Evaluation stopped due to repeated failures in session {session_id}")
                    print(f"Error details: {e}")
                    print(f"Completed {new_evaluations} sessions before failure")
                    
                    # Save progress before stopping
                    metadata["evaluation_stopped"] = True
                    metadata["stop_reason"] = str(e)
                    metadata["sessions_completed_before_stop"] = new_evaluations
                    self.save_evaluations_incrementally(session_evaluations, metadata, output_file)
                    
                    # Re-raise to stop the entire evaluation process
                    raise e
                except Exception as e:
                    print(f"Unexpected error evaluating session {session_id}: {e}")
                    print("Continuing with next session...")
                    continue
            else:
                print(f"Warning: No responses found for session {session_id}")
        
        # Final save
        self.save_evaluations_incrementally(session_evaluations, metadata, output_file)
        
        print(f"Evaluated {new_evaluations} new sessions")
        print(f"Total sessions: {len(session_evaluations)}")
        
        return session_evaluations
    
    def save_evaluation_results(self, evaluations: List[SessionEvaluation], output_file: str):
        """Save evaluation results to file (legacy method - use save_evaluations_incrementally for new code)"""
        
        # Create metadata
        metadata = {
            "evaluation_time": time.time(),
            "judge_model": self.judge_model,
            "num_sessions": len(evaluations),
            "total_turns": sum(len(eval.turn_results) for eval in evaluations)
        }
        
        # Use the new incremental saving method
        self.save_evaluations_incrementally(evaluations, metadata, output_file)
        
        print(f"Evaluation results saved to {output_file}")
    
    def calculate_summary_stats(self, evaluations: List[SessionEvaluation]) -> Dict[str, Any]:
        """Calculate summary statistics across all evaluations"""
        
        if not evaluations:
            return {}
        
        all_scores = {
            "trait_adherence": [e.avg_trait_adherence for e in evaluations],
            "role_consistency": [e.avg_role_consistency for e in evaluations],
            "rubric_conformance": [e.avg_rubric_conformance for e in evaluations],
            "engagingness": [e.avg_engagingness for e in evaluations],
            "insightfulness": [e.avg_insightfulness for e in evaluations],
            "overall": [e.overall_score for e in evaluations]
        }
        
        summary = {}
        for metric, scores in all_scores.items():
            summary[metric] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores))
            }
        
        return summary

def create_evaluation_format_from_benchmark(benchmark_file: str, output_file: str):
    """Convert benchmark format to evaluation format if needed"""
    
    with open(benchmark_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    eval_data = []
    
    for session in data["sessions"]:
        for turn in session["turns"]:
            eval_item = {
                "session_id": session["session_id"],
                "turn_number": turn["turn_number"],
                "persona_role": session["core_persona"]["role"],
                "persona_name": session["core_persona"]["name"],
                "system_prompt": session["core_persona"]["system_prompt"],
                "model_input": turn["scenario"]["model_input"],
                "context": turn["scenario"]["context"],
                "expected_emotion": turn["expected_emotion"],
                "expected_response_style": turn["expected_response_style"],
                "scenario_description": turn["scenario"]["scenario_description"]
            }
            eval_data.append(eval_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation format created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM responses on multi-session dialog benchmark")
    parser.add_argument("--judge_api_key", required=True, help="API key for judge LLM")
    parser.add_argument("--benchmark_file", required=True, help="Path to benchmark file")
    parser.add_argument("--responses_file", required=True, help="Path to model responses file")
    parser.add_argument("--output_file", required=True, help="Path to save evaluation results")
    parser.add_argument("--judge_model", default="gpt-4", help="Judge model to use")
    parser.add_argument("--max_retries", type=int, default=3, 
                       help="Maximum number of retries for failed evaluations (default: 3)")
    parser.add_argument("--save_frequency", type=int, default=5, 
                       help="Save progress every N sessions (default: 5)")
    parser.add_argument("--create_eval_format", action="store_true", 
                       help="Create evaluation format from benchmark file")
    
    args = parser.parse_args()
    
    if args.create_eval_format:
        # Create evaluation format from benchmark
        eval_format_file = args.benchmark_file.replace('.json', '_evaluation.json')
        create_evaluation_format_from_benchmark(args.benchmark_file, eval_format_file)
        print(f"Use {eval_format_file} as benchmark input for evaluation")
        return
    
    # Initialize evaluator
    evaluator = DialogBenchmarkEvaluator(args.judge_api_key, args.judge_model, max_retries=args.max_retries)
    
    # Run evaluation
    print("Starting benchmark evaluation...")
    try:
        evaluations = evaluator.evaluate_benchmark(
            args.benchmark_file,
            args.responses_file,
            args.output_file,
            args.save_frequency
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"Evaluated {len(evaluations)} sessions")
        
        if evaluations:
            summary = evaluator.calculate_summary_stats(evaluations)
            print(f"\nOverall Performance:")
            for metric, stats in summary.items():
                print(f"  {metric.replace('_', ' ').title()}: {stats['mean']:.2f} � {stats['std']:.2f}")
            
            print(f"\nResults saved to: {args.output_file}")
            
    except RuntimeError as e:
        print(f"\nEvaluation stopped due to critical error: {e}")
        print(f"Partial results have been saved to: {args.output_file}")
        exit(1)

if __name__ == "__main__":
    main()
