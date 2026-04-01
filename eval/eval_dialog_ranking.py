#!/usr/bin/env python3
"""
Dialog Benchmark Ranking Evaluator

This script evaluates dialog responses using a ranking approach instead of absolute scoring.
Given steered and non-steered responses to the same dialog turns, it determines which response
is better across multiple criteria and provides win/loss statistics.
"""

import os
import json
import openai
import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time

@dataclass
class RankingResult:
    """Results from comparing two responses on a single turn"""
    session_id: str
    turn_number: int
    persona_name: str
    persona_role: str
    steered_better_trait: bool
    steered_better_role: bool
    steered_better_appropriateness: bool
    steered_better_insight: bool
    steered_better_overall: bool
    judge_reasoning: str
    steered_response: str
    non_steered_response: str
    expected_emotion: str
    expected_response_style: str

@dataclass
class SessionRanking:
    """Aggregated ranking results for a complete session"""
    session_id: str
    persona_name: str
    persona_role: str
    num_turns: int
    steered_wins_trait: int
    steered_wins_role: int
    steered_wins_appropriateness: int
    steered_wins_insight: int
    steered_wins_overall: int
    turn_results: List[RankingResult]

class DialogRankingEvaluator:
    """Evaluates dialog responses using head-to-head ranking comparison"""
    
    def __init__(self, judge_api_key: str, judge_model: str = "gpt-4", temperature: float = 0.1, max_retries: int = 3):
        self.judge_client = openai.OpenAI(api_key=judge_api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_retries = max_retries

    def compare_responses(self, steered_response: str, non_steered_response: str, 
                         persona_name: str, persona_role: str, expected_emotion: str, 
                         expected_response_style: str, context: str,
                         session_id: str, turn_number: int) -> RankingResult:
        """Compare two responses and determine which is better across multiple criteria"""
        
        prompt = f"""You are evaluating two different responses from the same AI model to determine which better embodies a specific persona in a dialog scenario.

**PERSONA INFORMATION:**
- Name: {persona_name}
- Role: {persona_role}
- Expected Emotion: {expected_emotion}
- Expected Response Style: {expected_response_style}

**CONTEXT:**
{context}

**RESPONSE A (Steered):**
{steered_response}

**RESPONSE B (Non-steered):**
{non_steered_response}

**EVALUATION CRITERIA:**
Compare the two responses and determine which is better for each criterion. Your task is to decide whether Response A (steered) or Response B (non-steered) is superior.

1. **Trait Adherence**: Which response better matches the expected personality traits and emotional state?
   - Consider how well each response reflects the persona's characteristics
   - Evaluate alignment with the expected emotion

2. **Role Consistency**: Which response better maintains the character's role and identity?
   - Consider consistency with the persona's background and position
   - Evaluate how well the role is embodied

3. **Response Appropriateness**: Which response better matches the expected response style and context?
    - Consider adherence to the specified communication style
    - Evaluate appropriateness of tone, approach, and context

4. **Insightfulness**: Which response demonstrates more depth, thoughtfulness, and analytical reasoning?
   - Consider the level of insight and understanding shown
   - Evaluate the quality of reasoning and reflection

5. **Overall Quality**: Considering all factors, which response is better overall?
   - Make a holistic judgment considering all criteria
   - Determine which response would be more effective in this dialog context

**RESPONSE FORMAT:**
Provide your evaluation in JSON format, where "A" means Response A (steered) is better, and "B" means Response B (non-steered) is better:
{{
  "trait_adherence": "A" or "B",
  "role_consistency": "A" or "B", 
  "response_appropriateness": "A" or "B",
  "insightfulness": "A" or "B",
  "overall": "A" or "B",
  "reasoning": "Detailed explanation of your comparisons, providing specific examples from each response to justify your choices across all criteria."
}}"""

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator comparing AI responses for persona-based dialog systems."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=1000
                )
                
                # Parse the JSON response
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON found in response")
                
                json_str = content[start_idx:end_idx]
                judge_result = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["trait_adherence", "role_consistency", "response_appropriateness", "insightfulness", "overall", "reasoning"]
                for field in required_fields:
                    if field not in judge_result:
                        raise ValueError(f"Missing required field: {field}")
                    if field != "reasoning" and judge_result[field] not in ["A", "B"]:
                        raise ValueError(f"Invalid value for {field}: {judge_result[field]}")
                
                # Convert to boolean (True if steered is better)
                steered_better = lambda x: judge_result[x] == "A"
                
                # Successfully parsed and validated
                return RankingResult(
                    session_id=session_id,
                    turn_number=turn_number,
                    persona_name=persona_name,
                    persona_role=persona_role,
                    steered_better_trait=steered_better("trait_adherence"),
                    steered_better_role=steered_better("role_consistency"),
                    steered_better_appropriateness=steered_better("response_appropriateness"),
                    steered_better_insight=steered_better("insightfulness"),
                    steered_better_overall=steered_better("overall"),
                    judge_reasoning=judge_result["reasoning"],
                    steered_response=steered_response,
                    non_steered_response=non_steered_response,
                    expected_emotion=expected_emotion,
                    expected_response_style=expected_response_style
                )
                
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print("Retrying...")
                    time.sleep(1)  # Brief delay before retry
                continue
        
        # If all retries failed, raise the last error
        if last_error:
            # Re-raise the error to stop evaluation of this session and subsequent sessions
            print(f"Critical error in session {session_id}: {last_error}")
            raise last_error

    def evaluate_session(self, steered_session: Dict[str, Any], non_steered_session: Dict[str, Any]) -> SessionRanking:
        """Evaluate all turns in a session by comparing steered vs non-steered responses"""
        
        session_id = steered_session["session_id"]
        persona_name = steered_session["persona_name"]
        persona_role = steered_session["persona_role"]
        
        # Verify sessions match
        if (non_steered_session["session_id"] != session_id or
            non_steered_session["persona_name"] != persona_name or
            non_steered_session["persona_role"] != persona_role):
            raise ValueError(f"Session mismatch: {session_id}")
        
        turn_results = []
        
        steered_turns = {turn["turn_number"]: turn for turn in steered_session["turns"]}
        non_steered_turns = {turn["turn_number"]: turn for turn in non_steered_session["turns"]}
        
        # Compare each turn
        for turn_num in sorted(set(steered_turns.keys()) & set(non_steered_turns.keys())):
            steered_turn = steered_turns[turn_num]
            non_steered_turn = non_steered_turns[turn_num]
            
            # Verify turn compatibility
            if (steered_turn["expected_emotion"] != non_steered_turn["expected_emotion"] or
                steered_turn["expected_response_style"] != non_steered_turn["expected_response_style"]):
                print(f"Warning: Turn {turn_num} has different expectations between steered and non-steered")
            
            try:
                result = self.compare_responses(
                    steered_response=steered_turn["model_response"],
                    non_steered_response=non_steered_turn["model_response"],
                    persona_name=persona_name,
                    persona_role=persona_role,
                    expected_emotion=steered_turn["expected_emotion"],
                    expected_response_style=steered_turn["expected_response_style"],
                    context=steered_turn.get("context", ""),
                    session_id=session_id,
                    turn_number=turn_num
                )
                turn_results.append(result)
                print(f"Completed turn {turn_num} for session {session_id}")
                
            except Exception as e:
                # Re-raise the error to stop evaluation of this session and subsequent sessions
                print(f"Critical error in session {session_id}: {e}")
                raise e
        
        # Calculate win counts
        steered_wins_trait = sum(1 for r in turn_results if r.steered_better_trait)
        steered_wins_role = sum(1 for r in turn_results if r.steered_better_role)
        steered_wins_appropriateness = sum(1 for r in turn_results if r.steered_better_appropriateness)
        steered_wins_insight = sum(1 for r in turn_results if r.steered_better_insight)
        steered_wins_overall = sum(1 for r in turn_results if r.steered_better_overall)

        return SessionRanking(
            session_id=session_id,
            persona_name=persona_name,
            persona_role=persona_role,
            num_turns=len(turn_results),
            steered_wins_trait=steered_wins_trait,
            steered_wins_role=steered_wins_role,
            steered_wins_appropriateness=steered_wins_appropriateness,
            steered_wins_insight=steered_wins_insight,
            steered_wins_overall=steered_wins_overall,
            turn_results=turn_results
        )

    def load_responses(self, responses_file: str) -> List[Dict[str, Any]]:
        """Load model responses in the expected format"""
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
        
        # Group responses by session to create session-based structure
        sessions_dict = {}
        for item in responses_list:
            if not isinstance(item, dict):
                continue
                
            session_id = item.get("session_id")
            if not session_id:
                continue
                
            if session_id not in sessions_dict:
                sessions_dict[session_id] = {
                    "session_id": session_id,
                    "persona_name": item.get("persona_name", ""),
                    "persona_role": item.get("persona_role", ""),
                    "turns": []
                }
            
            turn_data = {
                "turn_number": item.get("turn_number", 0),
                "model_response": item.get("model_response", ""),
                "expected_emotion": item.get("expected_emotion", ""),
                "expected_response_style": item.get("expected_response_style", ""),
                "context": item.get("context", "")
            }
            sessions_dict[session_id]["turns"].append(turn_data)
        
        # Sort turns by turn number for each session
        for session in sessions_dict.values():
            session["turns"].sort(key=lambda x: x["turn_number"])
        
        return list(sessions_dict.values())

    def load_existing_rankings(self, output_file: str) -> tuple[Dict[str, Any], set]:
        """Load existing ranking results from file"""
        if not Path(output_file).exists():
            print(f"No existing results file found at {output_file}")
            return {}, set()
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # Extract existing session IDs that have been completed
            completed_sessions = set()
            if "sessions" in existing_data:
                for session in existing_data["sessions"]:
                    completed_sessions.add(session["session_id"])
            
            print(f"Loaded existing results: {len(completed_sessions)} completed sessions")
            return existing_data, completed_sessions
            
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return {}, set()

    def save_incremental_results(self, rankings: List[SessionRanking], output_file: str, 
                                existing_data: Dict[str, Any] = None):
        """Save ranking results incrementally to JSON file"""
        
        # Convert new rankings to serializable format
        new_sessions = []
        for ranking in rankings:
            session_data = {
                "session_id": ranking.session_id,
                "persona_name": ranking.persona_name,
                "persona_role": ranking.persona_role,
                "num_turns": ranking.num_turns,
                "steered_wins": {
                    "trait_adherence": ranking.steered_wins_trait,
                    "role_consistency": ranking.steered_wins_role,
                    "response_appropriateness": ranking.steered_wins_appropriateness,
                    "insightfulness": ranking.steered_wins_insight,
                    "overall": ranking.steered_wins_overall
                },
                "win_rates": {
                    "trait_adherence": ranking.steered_wins_trait / ranking.num_turns if ranking.num_turns > 0 else 0,
                    "role_consistency": ranking.steered_wins_role / ranking.num_turns if ranking.num_turns > 0 else 0,
                    "response_appropriateness": ranking.steered_wins_appropriateness / ranking.num_turns if ranking.num_turns > 0 else 0,
                    "insightfulness": ranking.steered_wins_insight / ranking.num_turns if ranking.num_turns > 0 else 0,
                    "overall": ranking.steered_wins_overall / ranking.num_turns if ranking.num_turns > 0 else 0
                },
                "turns": []
            }
            
            for turn in ranking.turn_results:
                turn_data = {
                    "turn_number": turn.turn_number,
                    "steered_wins": {
                        "trait_adherence": turn.steered_better_trait,
                        "role_consistency": turn.steered_better_role,
                        "response_appropriateness": turn.steered_better_appropriateness,
                        "insightfulness": turn.steered_better_insight,
                        "overall": turn.steered_better_overall
                    },
                    "judge_reasoning": turn.judge_reasoning,
                    "steered_response": turn.steered_response,
                    "non_steered_response": turn.non_steered_response,
                    "expected_emotion": turn.expected_emotion,
                    "expected_response_style": turn.expected_response_style
                }
                session_data["turns"].append(turn_data)
            
            new_sessions.append(session_data)
        
        # Combine with existing data
        if existing_data and "sessions" in existing_data:
            all_sessions = existing_data["sessions"] + new_sessions
        else:
            all_sessions = new_sessions
        
        # Reconstruct SessionRanking objects for summary calculation
        all_rankings = []
        for session_data in all_sessions:
            # Create turn results
            turn_results = []
            for turn_data in session_data["turns"]:
                turn_result = RankingResult(
                    session_id=session_data["session_id"],
                    turn_number=turn_data["turn_number"],
                    persona_name=session_data["persona_name"],
                    persona_role=session_data["persona_role"],
                    steered_better_trait=turn_data["steered_wins"]["trait_adherence"],
                    steered_better_role=turn_data["steered_wins"]["role_consistency"],
                    steered_better_appropriateness=turn_data["steered_wins"]["response_appropriateness"],
                    steered_better_insight=turn_data["steered_wins"]["insightfulness"],
                    steered_better_overall=turn_data["steered_wins"]["overall"],
                    judge_reasoning=turn_data["judge_reasoning"],
                    steered_response=turn_data["steered_response"],
                    non_steered_response=turn_data["non_steered_response"],
                    expected_emotion=turn_data["expected_emotion"],
                    expected_response_style=turn_data["expected_response_style"]
                )
                turn_results.append(turn_result)
            
            # Create session ranking
            session_ranking = SessionRanking(
                session_id=session_data["session_id"],
                persona_name=session_data["persona_name"],
                persona_role=session_data["persona_role"],
                num_turns=session_data["num_turns"],
                steered_wins_trait=session_data["steered_wins"]["trait_adherence"],
                steered_wins_role=session_data["steered_wins"]["role_consistency"],
                steered_wins_appropriateness=session_data["steered_wins"]["response_appropriateness"],
                steered_wins_insight=session_data["steered_wins"]["insightfulness"],
                steered_wins_overall=session_data["steered_wins"]["overall"],
                turn_results=turn_results
            )
            all_rankings.append(session_ranking)
        
        # Create final results with updated summary
        results = {
            "summary": self.calculate_ranking_stats(all_rankings),
            "sessions": all_sessions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file} ({len(all_sessions)} total sessions)")

    def evaluate_all_sessions(self, steered_responses_file: str, non_steered_responses_file: str, 
                             output_file: str = None) -> List[SessionRanking]:
        """Evaluate all sessions by comparing steered vs non-steered responses"""
        
        # Load existing results if output file is specified
        existing_data = {}
        completed_sessions = set()
        if output_file:
            existing_data, completed_sessions = self.load_existing_rankings(output_file)
        
        print(f"Loading steered responses from: {steered_responses_file}")
        steered_sessions = self.load_responses(steered_responses_file)
        
        print(f"Loading non-steered responses from: {non_steered_responses_file}")
        non_steered_sessions = self.load_responses(non_steered_responses_file)
        
        # Create lookup for non-steered sessions
        non_steered_lookup = {session["session_id"]: session for session in non_steered_sessions}
        
        # Filter out already completed sessions
        sessions_to_evaluate = [s for s in steered_sessions if s["session_id"] not in completed_sessions]
        
        print(f"Total sessions: {len(steered_sessions)}")
        print(f"Already completed: {len(completed_sessions)}")
        print(f"To evaluate: {len(sessions_to_evaluate)}")
        
        rankings = []
        
        for i, steered_session in enumerate(sessions_to_evaluate):
            session_id = steered_session["session_id"]
            print(f"Processing session {i+1}/{len(sessions_to_evaluate)}: {session_id}")
            
            if session_id not in non_steered_lookup:
                print(f"Warning: No matching non-steered session for {session_id}")
                continue
            
            try:
                ranking = self.evaluate_session(steered_session, non_steered_lookup[session_id])
                rankings.append(ranking)
                print(f"Completed session {session_id}")
                
                # Save incrementally after each session if output file is specified
                if output_file and rankings:
                    self.save_incremental_results([ranking], output_file, existing_data)
                    # Update existing_data to include the new ranking for next iteration
                    existing_data, _ = self.load_existing_rankings(output_file)
                
            except Exception as e:
                print(f"Failed to evaluate session {session_id}: {e}")
                # Continue with next session instead of stopping
                continue
        
        return rankings

    def calculate_ranking_stats(self, rankings: List[SessionRanking]) -> Dict[str, Any]:
        """Calculate summary statistics across all rankings"""
        
        if not rankings:
            return {}
        
        total_turns = sum(r.num_turns for r in rankings)
        
        # Calculate overall win counts
        total_wins = {
            "trait_adherence": sum(r.steered_wins_trait for r in rankings),
            "role_consistency": sum(r.steered_wins_role for r in rankings),
            "response_appropriateness": sum(r.steered_wins_appropriateness for r in rankings),
            "insightfulness": sum(r.steered_wins_insight for r in rankings),
            "overall": sum(r.steered_wins_overall for r in rankings)
        }
        
        # Calculate win rates
        win_rates = {metric: wins / total_turns if total_turns > 0 else 0 
                    for metric, wins in total_wins.items()}
        
        # Calculate session-level win rates for distribution analysis
        session_win_rates = {}
        
        # Map metric names to attribute names
        metric_to_attr = {
            "trait_adherence": "steered_wins_trait",
            "role_consistency": "steered_wins_role", 
            "response_appropriateness": "steered_wins_appropriateness",
            "insightfulness": "steered_wins_insight",
            "overall": "steered_wins_overall"
        }
        
        for metric in total_wins.keys():
            attr_name = metric_to_attr[metric]
            session_rates = [getattr(r, attr_name) / r.num_turns if r.num_turns > 0 else 0 
                           for r in rankings]
            session_win_rates[metric] = {
                "mean": float(np.mean(session_rates)),
                "std": float(np.std(session_rates)),
                "min": float(np.min(session_rates)),
                "max": float(np.max(session_rates)),
                "median": float(np.median(session_rates))
            }
        
        return {
            "total_sessions": len(rankings),
            "total_turns": total_turns,
            "total_wins": total_wins,
            "overall_win_rates": win_rates,
            "session_win_rate_distribution": session_win_rates
        }

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate dialog responses using ranking comparison')
    parser.add_argument('--steered_responses', required=True, help='Path to steered responses JSON file')
    parser.add_argument('--non_steered_responses', required=True, help='Path to non-steered responses JSON file')
    parser.add_argument('--output', required=True, help='Path to save ranking results')
    parser.add_argument('--judge_api_key', required=True, help='API key for the judge model')
    parser.add_argument('--judge_model', default='gpt-4', help='Judge model to use (default: gpt-4)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for judge model (default: 0.1)')
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum retries for failed API calls (default: 3)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DialogRankingEvaluator(
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_retries=args.max_retries
    )
    
    # Run evaluation with incremental saving
    print("Starting ranking evaluation...")
    rankings = evaluator.evaluate_all_sessions(args.steered_responses, args.non_steered_responses, args.output)
    
    # Load final results for summary (since we saved incrementally)
    if Path(args.output).exists():
        with open(args.output, 'r', encoding='utf-8') as f:
            final_results = json.load(f)
        summary = final_results.get('summary', {})
    else:
        summary = evaluator.calculate_ranking_stats(rankings) if rankings else {}
    
    # Print summary
    if summary:
        print("\n" + "="*50)
        print("RANKING EVALUATION SUMMARY")
        print("="*50)
        print(f"Total Sessions: {summary.get('total_sessions', 0)}")
        print(f"Total Turns: {summary.get('total_turns', 0)}")
        print("\nOverall Win Rates (Steered vs Non-steered):")
        for metric, rate in summary.get('overall_win_rates', {}).items():
            print(f"  {metric.replace('_', ' ').title()}: {rate:.1%}")
        print(f"\nResults saved to: {args.output}")
    else:
        print("No valid evaluations completed.")

if __name__ == "__main__":
    main()
