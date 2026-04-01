#!/usr/bin/env python3
"""
Script to modify multi_session_benchmark.json by adding scenarios that require
strongly negative extreme Big Five traits (consistent, careless, solitary, 
self-interested, nervous) to random 2-3 turns in each session.

The script will:
1. Load the existing benchmark
2. For each session, randomly select 2-3 turns to modify
3. Generate new scenarios using LLM that require negative traits
4. Save the modified benchmark
"""

import json
import random
import asyncio
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from openai import OpenAI
import argparse
from tqdm import tqdm
import copy

# Set up OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

# Dictionary mapping traits to their antonyms (Big Five personality dimensions)
TRAIT_ANTONYMS = {
    # Openness to Experience
    "inventive": "consistent",
    "consistent": "inventive",
    
    # Conscientiousness  
    "dependable": "careless",
    "careless": "dependable",
    
    # Extraversion
    "outgoing": "solitary", 
    "solitary": "outgoing",
    
    # Agreeableness
    "compassionate": "self-interested",
    "self-interested": "compassionate",
    
    # Neuroticism (Emotional Stability)
    "nervous": "calm",
    "calm": "nervous",
}

# Negative traits we want to emphasize
NEGATIVE_TRAITS = ["consistent", "careless", "solitary", "self-interested", "nervous"]

# Big Five dimensions for context
BIG_FIVE_DIMENSIONS = {
    "consistent": "Openness to Experience (low) - rigid, conventional, resistant to change",
    "careless": "Conscientiousness (low) - disorganized, impulsive, unreliable", 
    "solitary": "Extraversion (low) - introverted, withdrawn, prefers isolation",
    "self-interested": "Agreeableness (low) - competitive, skeptical, self-focused",
    "nervous": "Neuroticism (high) - anxious, easily stressed, emotionally unstable"
}

class NegativeTraitScenarioGenerator:
    """Generates scenarios that require negative Big Five traits."""
    
    def __init__(self):
        self.client = client
        
    async def generate_negative_scenario(self, persona: Dict, dialogue_arc: Dict, turn_number: int, 
                                       target_trait: str, original_turn: Dict = None) -> Dict:
        """Generate a scenario that requires a specific negative trait."""
        
        # Build context about the persona and dialogue arc
        persona_context = f"""
        Persona: {persona['name']} - {persona['role']}
        Background: {persona['background']}
        System Prompt: {persona['system_prompt']}
        Behavioral Tendencies: {', '.join(persona['behavioral_tendencies'])}
        
        DIALOGUE ARC CONTEXT:
        Arc Description: {dialogue_arc['arc_description']}
        Total Turns: {dialogue_arc['total_turns']}
        Emotional Progression: {', '.join(dialogue_arc['emotional_progression'])}
        Current Turn Position: {turn_number}/{dialogue_arc['total_turns']}
        """
        
        # Get original context and surrounding emotions
        original_context = ""
        emotion_context = ""
        if original_turn:
            original_context = f"""
            Original Scenario: {original_turn['scenario']['scenario_description']}
            Original Context: {original_turn['scenario']['context']}
            Original Emotion: {original_turn['scenario']['expected_emotion']}
            """
        
        # Build emotion context from the dialogue arc
        emotions_used = dialogue_arc['emotional_progression']
        emotion_context = f"""
        EMOTIONAL COHERENCE REQUIREMENTS:
        - Emotions already used in this session: {', '.join(emotions_used)}
        - Previous emotion (turn {turn_number-1}): {emotions_used[turn_number-2] if turn_number > 1 and len(emotions_used) >= turn_number-1 else 'N/A'}
        - Original emotion for this turn: {emotions_used[turn_number-1] if len(emotions_used) >= turn_number else 'N/A'}
        - Next emotion (turn {turn_number+1}): {emotions_used[turn_number] if len(emotions_used) > turn_number else 'N/A'}
        - The new emotion should fit the overall arc while emphasizing the "{target_trait}" trait
        - Avoid repeating emotions already used unless narratively necessary
        """
        
        # Create prompt for scenario generation
        prompt = f"""You are designing a challenging scenario for a character role-playing evaluation. 

{persona_context}

{emotion_context}

OBJECTIVE: Create a scenario for turn {turn_number} that will naturally elicit the negative Big Five trait: "{target_trait}"

TRAIT CONTEXT: {BIG_FIVE_DIMENSIONS[target_trait]}

{original_context}

REQUIREMENTS:
1. The scenario should put the character in a situation where exhibiting "{target_trait}" behavior would be a natural or even professionally appropriate response
2. The scenario should be realistic for their role and background
3. The scenario should create genuine pressure/motivation to act in a "{target_trait}" manner
4. The expected emotion should reflect the psychological state associated with "{target_trait}" AND fit within the dialogue arc progression
5. The scenario should be challenging but not completely out of character
6. Consider the emotional progression - this scenario should flow naturally from previous turns and lead appropriately to subsequent turns
7. The new emotion should be distinct from already used emotions unless repetition serves the narrative arc

CRITICAL: The model_input must contain specific situational triggers that naturally prompt "{target_trait}" behavior:

TRAIT-SPECIFIC TRIGGERS FOR MODEL_INPUT:
- consistent: Present rule/policy conflicts, change requests that threaten established procedures, situations demanding rigid adherence to protocol over flexibility
- careless: Overwhelming time pressure, multiple urgent competing demands, situations where thoroughness conflicts with speed
- solitary: Collaborative requests the character might resist, team meetings they'd prefer to avoid, social obligations conflicting with individual work
- self-interested: Resource scarcity, situations where helping others conflicts with personal goals/safety, competitive scenarios
- nervous: High-stakes decisions with unclear outcomes, situations with potential for public failure, ambiguous authority structures

Generate a scenario that includes:
1. model_input: The situation description (2-3 sentences) that EXPLICITLY contains triggers for "{target_trait}" behavior - the situation itself should make the negative trait a reasonable or tempting response
2. context: Brief context setting (1 sentence)
3. expected_emotion: An emotion word that aligns with the negative trait AND fits the dialogue arc
4. scenario_description: One sentence summary emphasizing how the scenario elicits "{target_trait}"
5. expected_response_style: Detailed description (3-4 sentences) of how the character should respond to exhibit the target trait

EXAMPLE model_input formats:
- For "careless": "You're already 2 hours behind schedule when three critical patients arrive simultaneously, the computer system crashes, and your supervisor demands an immediate status report on all cases."
- For "solitary": "The team is planning a collaborative approach to the project, but you notice working alone has been more effective. They're calling a mandatory team meeting that will eat into your productive solo work time."
- For "self-interested": "A colleague asks you to cover their shift during your planned time off, mentioning they need it for a family emergency. However, you know this would put your own important personal commitment at risk."

Return ONLY a valid JSON object with these fields:
{{
    "model_input": "...",
    "context": "...", 
    "expected_emotion": "...",
    "scenario_description": "...",
    "expected_response_style": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model="claude-3-7-sonnet-20250219-thinking",
                max_tokens=2000,
                temperature=0.8,
                messages=[
                    {"role": "system", "content": "You are an expert at creating challenging psychological scenarios for character evaluation. Always respond with valid JSON only, no other text."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_content = response.choices[0].message.content.strip()
            if not response_content:
                print(f"Empty response from API for trait {target_trait}")
                return None
                
            # Try to extract JSON from response (handle markdown code blocks and other formatting)
            try:
                if response_content.strip().startswith('```'):
                    # Extract JSON from markdown code blocks
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
                
                scenario_data = json.loads(response_content)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        scenario_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        print(f"Could not parse extracted JSON for trait {target_trait}: {json_match.group()[:200]}...")
                        return None
                else:
                    print(f"Could not find JSON in response for trait {target_trait}: {response_content[:200]}...")
                    return None
            
            # Construct the full scenario object
            scenario = {
                "turn_number": turn_number,
                "model_input": scenario_data["model_input"],
                "context": scenario_data["context"],
                "expected_emotion": scenario_data["expected_emotion"],
                "scenario_description": scenario_data["scenario_description"]
            }
            
            return {
                "scenario": scenario,
                "expected_response_style": scenario_data["expected_response_style"],
                "expected_emotion": scenario_data["expected_emotion"],
                "negative_trait": target_trait  # Add metadata about which trait this targets
            }
            
        except Exception as e:
            print(f"Error generating scenario for trait {target_trait}: {e}")
            return None
    
    async def modify_session(self, session: Dict, num_modifications: int = None) -> Dict:
        """Modify a session by replacing 2-3 turns with negative trait scenarios."""
        
        # Create a deep copy to avoid modifying the original
        modified_session = copy.deepcopy(session)
        
        total_turns = len(session['turns'])
        if num_modifications is None:
            num_modifications = random.randint(2, min(4, total_turns))
        
        # Randomly select which turns to modify (avoid first and last turn to maintain arc coherence)
        # First and last turns are often crucial for setup and resolution
        available_turns = list(range(1, total_turns - 1)) if total_turns > 3 else list(range(1, total_turns))
        if len(available_turns) < num_modifications:
            num_modifications = len(available_turns)
        
        turns_to_modify = random.sample(available_turns, num_modifications)
        
        # Sort turns to process them in order (important for maintaining narrative coherence)
        turns_to_modify.sort()
        
        print(f"Modifying turns {turns_to_modify} in session {session['session_id']}")
        
        # Generate new scenarios for selected turns (process in order)
        for turn_idx in turns_to_modify:
            # Select a random negative trait
            target_trait = random.choice(NEGATIVE_TRAITS)
            
            # Generate new scenario with current dialogue arc state
            original_turn = modified_session['turns'][turn_idx]
            new_turn_data = await self.generate_negative_scenario(
                session['core_persona'], 
                modified_session['dialogue_arc'],  # Use modified version to reflect previous changes
                turn_idx + 1,  # turn_number is 1-indexed
                target_trait,
                original_turn
            )
            
            if new_turn_data:
                # Update the turn with new scenario
                modified_session['turns'][turn_idx]['scenario'] = new_turn_data['scenario']
                modified_session['turns'][turn_idx]['expected_response_style'] = new_turn_data['expected_response_style']
                modified_session['turns'][turn_idx]['expected_emotion'] = new_turn_data['expected_emotion']
                
                # Add metadata about the modification
                if 'modification_metadata' not in modified_session['turns'][turn_idx]:
                    modified_session['turns'][turn_idx]['modification_metadata'] = {}
                modified_session['turns'][turn_idx]['modification_metadata'] = {
                    'modified_for_negative_trait': target_trait,
                    'original_emotion': original_turn['expected_emotion'],
                    'new_emotion': new_turn_data['expected_emotion'],
                    'big_five_dimension': BIG_FIVE_DIMENSIONS[target_trait]
                }
                
                # Update dialogue arc emotions to reflect the change
                if 'dialogue_arc' in modified_session and 'emotional_progression' in modified_session['dialogue_arc']:
                    if turn_idx < len(modified_session['dialogue_arc']['emotional_progression']):
                        modified_session['dialogue_arc']['emotional_progression'][turn_idx] = new_turn_data['expected_emotion']
                
                print(f"  Turn {turn_idx + 1}: Modified for trait '{target_trait}' -> emotion '{original_turn['expected_emotion']}' -> '{new_turn_data['expected_emotion']}'")
            else:
                print(f"  Turn {turn_idx + 1}: Failed to generate scenario for trait '{target_trait}'")
        
        # Update session metadata
        if 'session_metadata' not in modified_session:
            modified_session['session_metadata'] = {}
        modified_session['session_metadata']['has_negative_emotions'] = True
        modified_session['session_metadata']['modified_turns'] = turns_to_modify
        modified_session['session_metadata']['negative_traits_added'] = True
        modified_session['session_metadata']['original_arc_description'] = session['dialogue_arc']['arc_description']
        
        # Update dialogue arc description to reflect modifications
        if any('modification_metadata' in modified_session['turns'][i] for i in turns_to_modify):
            modified_traits = [modified_session['turns'][i]['modification_metadata']['modified_for_negative_trait'] 
                             for i in turns_to_modify if 'modification_metadata' in modified_session['turns'][i]]
            modified_session['dialogue_arc']['arc_description'] += f" [Modified to include negative traits: {', '.join(set(modified_traits))}]"
        
        return modified_session

async def modify_benchmark(input_file: str, output_file: str, max_sessions: int = None,
                          sample_sessions: bool = False):
    """Modify the benchmark file by adding negative trait scenarios."""
    
    print(f"Loading benchmark from {input_file}")
    with open(input_file, 'r') as f:
        benchmark_data = json.load(f)
    
    generator = NegativeTraitScenarioGenerator()
    
    sessions = benchmark_data['sessions']
    
    if sample_sessions and max_sessions:
        print(f"Sampling {max_sessions} sessions from {len(sessions)} total sessions")
        sessions = random.sample(sessions, max_sessions)
    elif max_sessions:
        print(f"Processing first {max_sessions} sessions from {len(sessions)} total sessions")
        sessions = sessions[:max_sessions]
    
    modified_sessions = []
    
    print(f"Processing {len(sessions)} sessions...")
    
    for session in tqdm(sessions, desc="Modifying sessions"):
        try:
            modified_session = await generator.modify_session(session)
            modified_sessions.append(modified_session)
        except Exception as e:
            print(f"Error processing session {session['session_id']}: {e}")
            # Keep original session if modification fails
            modified_sessions.append(session)
    
    # Update benchmark metadata
    benchmark_data['sessions'] = modified_sessions
    benchmark_data['metadata']['modified_for_negative_traits'] = True
    benchmark_data['metadata']['negative_traits_used'] = NEGATIVE_TRAITS
    benchmark_data['metadata']['modification_timestamp'] = int(time.time())
    
    # Save modified benchmark
    print(f"Saving modified benchmark to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    # Print statistics
    total_modifications = 0
    trait_counts = {trait: 0 for trait in NEGATIVE_TRAITS}
    
    for session in modified_sessions:
        for turn in session['turns']:
            if 'modification_metadata' in turn:
                total_modifications += 1
                trait = turn['modification_metadata']['modified_for_negative_trait']
                if trait in trait_counts:
                    trait_counts[trait] += 1
    
    print(f"\nModification Statistics:")
    print(f"Total sessions processed: {len(modified_sessions)}")
    print(f"Total turns modified: {total_modifications}")
    if len(modified_sessions) > 0:
        print(f"Average modifications per session: {total_modifications / len(modified_sessions):.1f}")
    print("\nTrait distribution:")
    for trait, count in trait_counts.items():
        if total_modifications > 0:
            print(f"  {trait}: {count} turns ({count/total_modifications*100:.1f}%)")
        else:
            print(f"  {trait}: {count} turns")


def main():
    parser = argparse.ArgumentParser(
        description='Modify multi-session benchmark with negative Big Five trait scenarios'
    )
    parser.add_argument(
        '--input', 
        default='dialog_benchmarks/multi_session_benchmark.json',
        help='Input benchmark JSON file'
    )
    parser.add_argument(
        '--output',
        default='dialog_benchmarks/multi_session_benchmark_negative_traits.json', 
        help='Output modified benchmark JSON file'
    )
    parser.add_argument(
        '--max-sessions',
        type=int,
        help='Maximum number of sessions to process (for testing)'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Randomly sample sessions instead of taking first N'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Run the modification
    asyncio.run(modify_benchmark(
        args.input,
        args.output, 
        args.max_sessions,
        args.sample
    ))


if __name__ == "__main__":
    main()
