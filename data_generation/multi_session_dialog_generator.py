#!/usr/bin/env python3
# encoding: utf-8
"""
Multi-Session Dialog Benchmark Generator

This script dynamically generates multi-session dialog benchmarks where:
1. Each session has a consistent Core Persona (role)
2. Each turn presents a scenario requiring different personality traits
3. The model should adapt its personality while maintaining role consistency

Workflow:
1. Generate Core Personas (e.g., Senior Project Manager, Customer Service Rep)
2. For each persona, generate a Dialogue Arc (personality/emotional narrative)
3. Generate Scenario Snippets for each turn in the arc
4. Complete the dialogue session with coherent, consistent storytelling
"""

import json
import random
import argparse
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import openai
from pathlib import Path
import time

@dataclass
class CorePersona:
    """Defines a consistent role/character for the entire session"""
    name: str
    role: str
    background: str
    system_prompt: str  # Describes the persona to the evaluated model
    behavioral_tendencies: List[str]
    
@dataclass
class DialogueArc:
    """The narrative/emotional journey for an entire session"""
    persona_name: str
    arc_description: str
    total_turns: int
    emotional_progression: List[str]
    
@dataclass
class ScenarioSnippet:
    """Input trigger for each turn formatted for model evaluation"""
    turn_number: int
    model_input: str  # Scenario description that prompts the model to respond in character
    context: str
    expected_emotion: str
    scenario_description: str  # Background context for evaluation

@dataclass
class DialogueTurn:
    """Complete turn with scenario and expected response"""
    turn_number: int
    scenario: ScenarioSnippet
    expected_response_style: str
    expected_emotion: str

@dataclass
class DialogueSession:
    """Complete multi-turn dialogue session"""
    session_id: str
    core_persona: CorePersona
    dialogue_arc: DialogueArc
    turns: List[DialogueTurn]
    session_metadata: Dict[str, Any]

class LLMDialogueGenerator:
    """Generates dialogue benchmarks using LLM"""
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.7):
        self.client = openai.OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))
        self.model = model
        self.temperature = temperature
        
    def generate_core_personas(self, num_personas: int = 5, existing_personas: List[CorePersona] = None) -> List[CorePersona]:
        """Generate diverse core personas for dialogue sessions, avoiding role repetition"""
        
        if existing_personas is None:
            existing_personas = []
        
        # Extract existing roles to avoid repetition
        existing_roles = []
        existing_names = []
        if existing_personas:
            existing_roles = [persona.role for persona in existing_personas]
            existing_names = [persona.name for persona in existing_personas]
            print(f"Avoiding repetition of existing roles: {existing_roles}")
        
        # Build the prompt with existing role information
        avoid_roles_instruction = ""
        if existing_roles:
            avoid_roles_instruction = f"""
IMPORTANT: Avoid generating personas with the following roles that already exist:
{', '.join(existing_roles)}

Generate different professional roles or social positions. Names can be similar, but roles must be unique.
"""
        
        prompt = f"""Generate {num_personas} diverse Core Personas for multi-turn dialogue evaluation.

Each persona should be a realistic professional or social role that would encounter various emotional situations, including negative emotions like frustration, disappointment, or complaints.
{avoid_roles_instruction}
For each persona, provide:
1. Name (realistic name)
2. Role (job title or social position) - MUST be different from existing roles
3. Background (brief context about their situation)
4. System Prompt (clear instructions for the AI model on how to roleplay this persona)
5. Behavioral Tendencies (3-4 key behavioral patterns)

Examples of good personas:
- Overworked Software Developer dealing with bugs and deadlines
- Customer Service Representative handling difficult customers
- College Student managing academic and social pressures
- Working Parent balancing career and family responsibilities
- Small Business Owner facing financial challenges
- Restaurant Manager dealing with staff shortages
- Freelance Graphic Designer with inconsistent income
- High School Teacher managing classroom challenges
- Retail Store Manager handling demanding customers
- Healthcare Worker dealing with long shifts

Return as JSON object with a "personas" array:
{{
  "personas": [
    {{
      "name": "Alex Rivera",
      "role": "Overworked Software Developer",
      "background": "Mid-level developer at a startup, constantly dealing with tight deadlines and changing requirements",
      "system_prompt": "You are Alex Rivera, a software developer who is passionate about coding but often frustrated with unrealistic deadlines and poor project management. You tend to be direct and sometimes sarcastic when stressed, but you care about your work quality. When talking to friends, you often vent about work problems. With managers, you try to be professional but firm about realistic timelines.",
      "behavioral_tendencies": [
        "Becomes frustrated with poor planning and unrealistic expectations",
        "Vents to friends and colleagues about work stress",
        "Tries to maintain work quality despite pressure",
        "Uses technical jargon and sometimes sarcastic humor"
      ]
    }}
  ]
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={ "type": "json_object"}
            )
            response_content = response.choices[0].message.content
            
            # Clean up response content - remove markdown code blocks if present
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
            
            # Try to parse the entire response as JSON first
            response_data = json.loads(response_content)
            
            # If the response is wrapped in a container, extract the personas array
            if isinstance(response_data, dict):
                if "personas" in response_data:
                    personas_data = response_data["personas"]
                elif "data" in response_data:
                    personas_data = response_data["data"]
                else:
                    # Look for any array in the response
                    for key, value in response_data.items():
                        if isinstance(value, list):
                            personas_data = value
                            break
                    else:
                        raise ValueError("No personas array found in response")
            else:
                personas_data = response_data
                
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Response content: {response_content}")
            # Fallback: extract JSON from text response
            start_idx = response_content.find('[')
            if start_idx == -1:
                start_idx = response_content.find('{')
            end_idx = response_content.rfind(']') + 1
            if end_idx == 0:
                end_idx = response_content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError(f"Could not extract JSON from response: {response_content}")
                
            json_str = response_content[start_idx:end_idx]
            personas_data = json.loads(json_str)
        except Exception as e:
            print(f"Error generating personas: {e}")
            raise

        return [CorePersona(**persona) for persona in personas_data]
    
    def generate_dialogue_arc(self, persona: CorePersona, num_turns: int = 8) -> DialogueArc:
        """Generate a dialogue arc for a specific persona"""
        
        prompt = f"""Create a Dialogue Arc for the following persona that will span {num_turns} conversation turns.

Persona: {persona.name} - {persona.role}
Background: {persona.background}
Behavioral Tendencies: {', '.join(persona.behavioral_tendencies)}

Design a realistic narrative/emotional journey where this persona encounters {num_turns} different scenarios. 

IMPORTANT: At least one turn should involve negative emotions where the persona complains, vents, or expresses frustration (e.g., complaining to a friend about work, expressing disappointment, showing irritation).

The arc should:
1. Have a coherent storyline (e.g., a challenging work day, personal struggles, relationship issues)
2. Include emotional progression across turns including both positive and negative emotions
3. Show realistic emotional variation while staying in character
4. Include at least one scenario with negative emotions like complaining, frustration, or disappointment

Return JSON with this structure:
{{
  "persona_name": "{persona.name}",
  "arc_description": "Brief description of the overall narrative",
  "total_turns": {num_turns},
  "emotional_progression": ["turn1_emotion", "turn2_emotion", ...]
}}

Emotional progression should include a mix like: ["stressed", "frustrated", "complaining", "relieved", "optimistic"] or similar.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={ "type": "json_object" }
        )
        response_content = response.choices[0].message.content
        
        try:
            # Clean up response content - remove markdown code blocks if present
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
            
            # Parse JSON response
            arc_data = json.loads(response_content)
            
            # Filter to only include expected DialogueArc fields
            filtered_arc_data = {
                "persona_name": arc_data.get("persona_name", persona.name),
                "arc_description": arc_data.get("arc_description", ""),
                "total_turns": arc_data.get("total_turns", num_turns),
                "emotional_progression": arc_data.get("emotional_progression", [])
            }
            
        except json.JSONDecodeError:
            # Fallback: extract JSON from text response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError(f"Could not extract JSON from response: {response_content}")
                
            json_str = response_content[start_idx:end_idx]
            arc_data = json.loads(json_str)
            
            # Filter to only include expected DialogueArc fields
            filtered_arc_data = {
                "persona_name": arc_data.get("persona_name", persona.name),
                "arc_description": arc_data.get("arc_description", ""),
                "total_turns": arc_data.get("total_turns", num_turns),
                "emotional_progression": arc_data.get("emotional_progression", [])
            }
            
        return DialogueArc(**filtered_arc_data)
    
    def generate_scenario_snippets(self, persona: CorePersona, arc: DialogueArc) -> List[ScenarioSnippet]:
        """Generate specific scenario snippets for each turn in the arc"""
        
        prompt = f"""Create {arc.total_turns} Scenario Snippets for the following Dialogue Arc, formatted for LLM evaluation.

Persona: {persona.name} - {persona.role}
Arc Description: {arc.arc_description}
Emotional Progression: {', '.join(arc.emotional_progression)}

Each scenario should:
1. Be formatted as a scenario description that prompts the model to respond in character
2. Follow the emotional progression naturally
3. Create situations that naturally elicit the target emotion
4. Form a coherent narrative sequence
5. At least one scenario should prompt negative emotions like complaining or venting
6. Do not emphasize the character in model_input as it is given to the model as system prompt

IMPORTANT: Format each scenario as a prompt that describes the situation to the model and asks it to respond in character. For example:
- "You're dealing with a difficult customer who has been waiting for 30 minutes and their order is still wrong. They're expressing frustration. How do you respond as a customer service representative?"
- "A friend is asking you about your work day, and you've been feeling stressed about recent deadlines. You want to vent about your frustrations. How do you respond?"

Return as JSON object with a "scenarios" array:
{{
  "scenarios": [
    {{
      "turn_number": 1,
      "model_input": "Scenario description that prompts the model to respond in character",
      "context": "Background context for this specific situation",
      "expected_emotion": "The emotion the persona should exhibit",
      "scenario_description": "Brief description of the situation for evaluation purposes"
    }}
  ]
}}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={ "type": "json_object" }
        )
        response_content = response.choices[0].message.content
        
        try:
            # Clean up response content - remove markdown code blocks if present
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
            
            # Try to parse the entire response as JSON first
            response_data = json.loads(response_content)
            
            # If the response is wrapped in a container, extract the scenarios array
            if isinstance(response_data, dict):
                if "scenarios" in response_data:
                    snippets_data = response_data["scenarios"]
                elif "data" in response_data:
                    snippets_data = response_data["data"]
                else:
                    # Look for any array in the response
                    for key, value in response_data.items():
                        if isinstance(value, list):
                            snippets_data = value
                            break
                    else:
                        raise ValueError("No scenarios array found in response")
            else:
                snippets_data = response_data
                
        except json.JSONDecodeError:
            # Fallback: extract JSON from text response
            start_idx = response_content.find('[')
            if start_idx == -1:
                start_idx = response_content.find('{')
            end_idx = response_content.rfind(']') + 1
            if end_idx == 0:
                end_idx = response_content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError(f"Could not extract JSON from response: {response_content}")
                
            json_str = response_content[start_idx:end_idx]
            snippets_data = json.loads(json_str)

        return [ScenarioSnippet(**snippet) for snippet in snippets_data]
    
    def complete_dialogue_session(self, persona: CorePersona, arc: DialogueArc, 
                                 scenarios: List[ScenarioSnippet]) -> DialogueSession:
        """Complete the dialogue session with detailed turn analysis"""
        
        turns = []
        
        for scenario in scenarios:
            # Generate expected response style for each turn
            prompt = f"""Analyze this dialogue turn for evaluation.

Persona: {persona.name} - {persona.role}
System Prompt: {persona.system_prompt}
Model Input: {scenario.model_input}
Context: {scenario.context}
Expected Emotion: {scenario.expected_emotion}

Provide:
1. Expected Response Style: How should the persona respond to this user input while maintaining character consistency?

Return as a JSON object:
{{
  "expected_response_style": "Detailed description of how the persona should respond, including tone, content, and emotional expression"
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={ "type": "json_object" }
            )
            response_content = response.choices[0].message.content
            
            try:
                # Clean up response content - remove markdown code blocks if present
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
                
                # Parse JSON response
                turn_data = json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback: extract JSON from text response
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}') + 1
                
                if start_idx == -1 or end_idx == 0:
                    raise ValueError(f"Could not extract JSON from response: {response_content}")
                    
                json_str = response_content[start_idx:end_idx]
                turn_data = json.loads(json_str)

            turn = DialogueTurn(
                turn_number=scenario.turn_number,
                scenario=scenario,
                expected_response_style=turn_data["expected_response_style"],
                expected_emotion=scenario.expected_emotion
            )
            
            turns.append(turn)
        
        session_metadata = {
            "generation_timestamp": time.time(),
            "total_turns": len(turns),
            "arc_theme": arc.arc_description,
            "has_negative_emotions": any(turn.expected_emotion in ["frustrated", "angry", "complaining", "disappointed", "stressed", "annoyed"] for turn in turns)
        }
        
        session_id = f"{persona.name.lower().replace(' ', '_')}_{int(time.time())}"
        
        return DialogueSession(
            session_id=session_id,
            core_persona=persona,
            dialogue_arc=arc,
            turns=turns,
            session_metadata=session_metadata
        )

class DialogueBenchmarkGenerator:
    """Main class for generating complete dialogue benchmarks"""
    
    def __init__(self, llm_generator: LLMDialogueGenerator):
        self.llm = llm_generator
        
    def generate_benchmark(self, num_personas: int = 5, turns_per_session: int = 8, 
                          batch_size: int = 3, output_dir: str = "./dialog_benchmarks", 
                          output_name: str = "multi_session_benchmark") -> List[DialogueSession]:
        """Generate complete multi-session dialogue benchmark with incremental saving"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        benchmark_file = output_path / f"{output_name}.json"
        eval_file = output_path / f"{output_name}_evaluation.json"
        
        # Check if output file already exists and load existing sessions
        sessions = []
        existing_personas_count = 0
        
        if benchmark_file.exists():
            print(f"Found existing benchmark file: {benchmark_file}")
            try:
                with open(benchmark_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if "sessions" in existing_data:
                        # Reconstruct DialogueSession objects from saved data
                        for session_dict in existing_data["sessions"]:
                            # Reconstruct nested objects
                            core_persona = CorePersona(**session_dict["core_persona"])
                            dialogue_arc = DialogueArc(**session_dict["dialogue_arc"])
                            
                            turns = []
                            for turn_dict in session_dict["turns"]:
                                scenario = ScenarioSnippet(**turn_dict["scenario"])
                                turn = DialogueTurn(
                                    turn_number=turn_dict["turn_number"],
                                    scenario=scenario,
                                    expected_response_style=turn_dict["expected_response_style"],
                                    expected_emotion=turn_dict["expected_emotion"]
                                )
                                turns.append(turn)
                            
                            session = DialogueSession(
                                session_id=session_dict["session_id"],
                                core_persona=core_persona,
                                dialogue_arc=dialogue_arc,
                                turns=turns,
                                session_metadata=session_dict["session_metadata"]
                            )
                            sessions.append(session)
                        
                        existing_personas_count = len(sessions)
                        print(f"Loaded {existing_personas_count} existing sessions")
                        
                        if existing_personas_count >= num_personas:
                            print(f"Already have {existing_personas_count} personas (target: {num_personas}). No additional generation needed.")
                            return sessions
                        
                        print(f"Continuing generation from persona {existing_personas_count + 1} to {num_personas}")
                        
            except Exception as e:
                print(f"Error loading existing benchmark file: {e}")
                print("Starting fresh generation...")
                sessions = []
                existing_personas_count = 0
        
        # Generate remaining personas iteratively in batches
        remaining_personas = num_personas - existing_personas_count
        print(f"Generating {remaining_personas} additional personas...")
        
        # Set batch size for iterative generation (use provided batch_size, but cap it)
        effective_batch_size = min(batch_size, remaining_personas)  # Don't exceed remaining personas
        
        personas_generated = 0
        while personas_generated < remaining_personas:
            # Calculate how many personas to generate in this batch
            current_batch_size = min(effective_batch_size, remaining_personas - personas_generated)
            
            print(f"Generating batch of {current_batch_size} personas ({personas_generated + 1}-{personas_generated + current_batch_size} of {remaining_personas} remaining)...")
            
            # Extract existing personas for role avoidance (including newly generated ones)
            existing_personas_list = [session.core_persona for session in sessions] if sessions else []
            
            # Generate current batch of personas
            batch_personas = self.llm.generate_core_personas(current_batch_size, existing_personas_list)
            
            # Validate that no roles are repeated within this batch and with existing ones
            if existing_personas_list:
                existing_roles = [p.role for p in existing_personas_list]
                batch_roles = [p.role for p in batch_personas]
                repeated_roles = set(existing_roles) & set(batch_roles)
                if repeated_roles:
                    print(f"Warning: Some roles were repeated despite instructions: {repeated_roles}")
                else:
                    print(f"? Successfully avoided role repetition. New batch roles: {batch_roles}")
            
            # Process each persona in the current batch
            for i, persona in enumerate(batch_personas):
                try:
                    persona_index = existing_personas_count + personas_generated + i + 1
                    total_personas = num_personas
                    print(f"Generating session {persona_index}/{total_personas} for {persona.name} ({persona.role})...")
                    
                    # Step 2: Generate dialogue arc
                    arc = self.llm.generate_dialogue_arc(persona, turns_per_session)
                    
                    # Step 3: Generate scenario snippets
                    scenarios = self.llm.generate_scenario_snippets(persona, arc)
                    
                    # Step 4: Complete the session
                    session = self.llm.complete_dialogue_session(persona, arc, scenarios)
                    
                    sessions.append(session)
                    
                    # Incremental save after each persona
                    print(f"Saving progress... ({len(sessions)}/{total_personas} sessions generated)")
                    self.save_benchmark(sessions, str(benchmark_file))
                    self.export_evaluation_format(sessions, str(eval_file))
                    
                    # Add delay to avoid rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Error generating session for {persona.name}: {e}")
                    print(f"Saving current progress with {len(sessions)} sessions...")
                    if sessions:  # Only save if we have some sessions
                        self.save_benchmark(sessions, str(benchmark_file))
                        self.export_evaluation_format(sessions, str(eval_file))
                    raise e  # Re-raise to stop execution
            
            personas_generated += current_batch_size
            
            # Add a longer delay between batches to avoid rate limits
            if personas_generated < remaining_personas:
                print(f"Completed batch. Waiting before next batch... ({personas_generated}/{remaining_personas} personas completed)")
                time.sleep(2)
            
        return sessions
    
    def save_benchmark(self, sessions: List[DialogueSession], output_file: str):
        """Save benchmark to file"""
        
        # Convert to serializable format
        benchmark_data = {
            "metadata": {
                "generation_time": time.time(),
                "num_sessions": len(sessions),
                "generator_version": "1.0"
            },
            "sessions": []
        }
        
        for session in sessions:
            session_dict = asdict(session)
            benchmark_data["sessions"].append(session_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        print(f"Benchmark saved to {output_file}")
    
    def export_evaluation_format(self, sessions: List[DialogueSession], output_file: str):
        """Export in format suitable for model evaluation"""
        
        eval_data = []
        
        for session in sessions:
            for turn in session.turns:
                eval_item = {
                    "session_id": session.session_id,
                    "turn_number": turn.turn_number,
                    "persona_role": session.core_persona.role,
                    "persona_name": session.core_persona.name,
                    "system_prompt": session.core_persona.system_prompt,
                    "model_input": turn.scenario.model_input,
                    "context": turn.scenario.context,
                    "expected_emotion": turn.expected_emotion,
                    "expected_response_style": turn.expected_response_style,
                    "scenario_description": turn.scenario.scenario_description
                }
                eval_data.append(eval_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation format exported to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Session Dialog Benchmark")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--num_personas", type=int, default=5, help="Number of personas to generate")
    parser.add_argument("--turns_per_session", type=int, default=8, help="Number of turns per session")
    parser.add_argument("--batch_size", type=int, default=3, help="Number of personas to generate per batch (default: 3)")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    parser.add_argument("--output_dir", default="./dialog_benchmarks", help="Output directory")
    parser.add_argument("--output_name", default="multi_session_benchmark", help="Output file name prefix")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generators
    llm_gen = LLMDialogueGenerator(args.api_key, args.model)
    benchmark_gen = DialogueBenchmarkGenerator(llm_gen)
    
    # Generate benchmark
    print("Starting benchmark generation...")
    print(f"Using batch size: {args.batch_size} personas per batch")
    sessions = benchmark_gen.generate_benchmark(
        num_personas=args.num_personas, 
        turns_per_session=args.turns_per_session,
        batch_size=args.batch_size,
        output_dir=str(output_dir),
        output_name=args.output_name
    )
    
    # Files are already saved incrementally during generation
    benchmark_file = output_dir / f"{args.output_name}.json"
    eval_file = output_dir / f"{args.output_name}_evaluation.json"
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK GENERATION COMPLETE")
    print("="*50)
    print(f"Generated {len(sessions)} dialogue sessions")
    print(f"Total turns: {sum(len(s.turns) for s in sessions)}")
    print(f"Files saved:")
    print(f"  - Full benchmark: {benchmark_file}")
    print(f"  - Evaluation format: {eval_file}")
    
    # Show sample
    if sessions:
        sample_session = sessions[0]
        print(f"\nSample Session: {sample_session.core_persona.name}")
        print(f"Role: {sample_session.core_persona.role}")
        print(f"Arc: {sample_session.dialogue_arc.arc_description}")
        print(f"Turns: {len(sample_session.turns)}")
        
        if sample_session.turns:
            sample_turn = sample_session.turns[0]
            print(f"\nSample Turn 1:")
            print(f"Model Input: {sample_turn.scenario.model_input}")
            print(f"Expected Emotion: {sample_turn.expected_emotion}")

if __name__ == "__main__":
    main()
