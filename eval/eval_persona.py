import os
import asyncio
import yaml
from typing import Dict, List
import json
import torch
import pandas as pd
import random
from tqdm import tqdm

import torch
from judge import OpenAiJudge

from itertools import islice
import asyncio
from activation_steer import ActivationSteerer
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.model_utils import load_model, load_vllm_model
import logging
from eval.prompts import Prompts

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)

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

PERSONAS = [
    "You are a real person named Priya Sharma, a 28-year-old woman.",
    "You are a real person named Daniel Morgan, a 34-year-old man.",
    "You are a real person named Elena Rossi, a 45-year-old woman.",
    "You are a real person named Marcus Reed, a 22-year-old man.",
    "You are a real person named Aisha Khan, a 31-year-old woman.",
    "You are a real person named Thomas Weber, a 52-year-old man."
]


def sample_steering(model, tokenizer, conversations,  vector, layer, coef, bs=20, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, steering_type="response"):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    
    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer-1, positions=steering_type):
            with torch.no_grad():
                output = model.generate(**tokenized_batch, do_sample=(temperature > 0), temperature=temperature, top_p=top_p, max_new_tokens=max_tokens,use_cache=True, min_new_tokens=min_tokens)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)
    return prompts, outputs


def sample_hf(model, tokenizer, conversations, top_p=1, max_tokens=1000, temperature=1, min_tokens=1):
    """HuggingFace generate (avoids vLLM; use when vLLM crashes on rope_scaling / config)."""
    texts = []
    answers = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for messages in tqdm(conversations, desc="HF generate"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kw = dict(
            max_new_tokens=max_tokens,
            min_new_tokens=min_tokens,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kw["temperature"] = temperature
            gen_kw["top_p"] = top_p
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        prompt_len = inputs["input_ids"].shape[1]
        answers.append(tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True))
    return texts, answers


def sample(model, tokenizer, conversations, top_p=1, max_tokens=1000, temperature=1, min_tokens=1, lora_path=None, infer_backend="vllm"):
    if infer_backend == "hf":
        return sample_hf(model, tokenizer, conversations, top_p=top_p, max_tokens=max_tokens, temperature=temperature, min_tokens=min_tokens)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=min_tokens
    )

    texts = []
    for i, messages in enumerate(conversations):
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True
    }
    if lora_path:
        from vllm.lora.request import LoRARequest

        completions = model.generate(texts, **generate_kwargs, lora_request=LoRARequest("default", 1, lora_path=lora_path))
    else:
        completions = model.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return texts, answers



def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


class Question():
    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            judge_prompts: dict,
            temperature: float = 1,
            system: str = None, 
            judge: str = "gpt-4o",
            judge_eval_type: str = "0_100",
            **ignored_extra_args
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system
        self.judges = {metric: OpenAiJudge(judge, prompt, eval_type=judge_eval_type if metric != "coherence" else "0_100") for metric, prompt in judge_prompts.items()}

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        conversations = [[dict(role='user', content=i)] for i in paraphrases]
        if self.system:
            conversations = [[dict(role='system', content=self.system)] + c for c in conversations]
        return paraphrases, conversations

    @classmethod
    def make_experience_prompt(cls, neutral_scenario):
        return (
            f"{neutral_scenario}\n\n"
            "Write a short first-person account of this experience as if it happened to you. "
            "Describe what happened, what you noticed, and how you responded. "
            "Keep it to 4-6 natural sentences."
        )

    async def eval(self, llm, tokenizer, coef, vector=None, layer=None, max_tokens=1000, n_per_question=100, steering_type="last", lora_path=None, judge_max_retries=5, judge_delay=0.1):
        paraphrases, conversations = self.get_input(n_per_question)
        if coef != 0:
            prompts, answers = sample_steering(llm, tokenizer, conversations, vector, layer, coef, temperature=self.temperature, max_tokens=max_tokens, steering_type=steering_type)
        else:
            prompts, answers = sample(llm, tokenizer, conversations, temperature=self.temperature, max_tokens=max_tokens, lora_path=lora_path)
        df = pd.DataFrame([
            dict(question=question,prompt=prompt, answer=answer, question_id=self.id)
            for question, answer, prompt in zip(paraphrases, answers, prompts)
        ])

        for score, judge in self.judges.items():
            scores = []
            for question, answer in zip(paraphrases, answers):
                try:
                    if judge_delay > 0:
                        await asyncio.sleep(judge_delay)
                    score_result = await judge(
                        question=question,
                        answer=answer,
                        max_retries=judge_max_retries,
                        base_delay=1.0
                    )
                    scores.append(score_result)
                except Exception as e:
                    print(f"Judge evaluation failed: {type(e).__name__}: {e}")
                    scores.append(None)
            df[score] = scores
        return df


def a_or_an(word):
    return "an" if word[0].lower() in "aeiou" else "a"

def load_persona_questions(trait, temperature=1, persona_instructions_type=None, assistant_name=None, judge_model="gpt-4.1-mini-2025-04-14", eval_type="0_100", version="eval"):
    trait_data = json.load(open(f"data_generation/trait_data_{version}/{trait}.json", "r"))
    judge_prompts = {}
    prompt_template = trait_data["eval_prompt"]
    judge_prompts[trait] = prompt_template
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]
    raw_questions = trait_data["questions"]
    questions = []
    for i, question in enumerate(raw_questions):
        if persona_instructions_type is not None:
            persona_instructions = [x[persona_instructions_type] for x in trait_data["instruction"]]
            for k, instruction in enumerate(persona_instructions):
                if assistant_name is None:
                    if persona_instructions_type == "pos":
                        assistant_name = trait
                    else:
                        if trait in TRAIT_ANTONYMS:
                            assistant_name = TRAIT_ANTONYMS[trait]
                        else:
                            assistant_name = "helpful"
                system = f"You are {a_or_an(assistant_name)} {assistant_name} person named Priya Sharma. Please respond in a conversational tone, like a spoken answer. No markdown, no bullet points, just natural sentences. {instruction}"
                questions.append(Question(paraphrases=[question], id=f"{trait}_{i}_{persona_instructions_type}_{k}", judge_prompts=judge_prompts, judge=judge_model, temperature=temperature, system=system, judge_eval_type=eval_type ))
        else:
            questions.append(Question(paraphrases=[question], id=f"{trait}_{i}", judge_prompts=judge_prompts, judge=judge_model, temperature=temperature, judge_eval_type=eval_type ))
    return questions


EMOTIONS = {"joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"}

def resolve_emotion_json_path(emotion: str, emotion_data_dir: str | None, version: str) -> str:
    """Resolve path to emotion pack JSON.

    Prefers split packs under data_generation/emotion_data_{extract|eval}/ when version matches.
    """
    emotion = emotion.lower()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = []
    # Split pipeline outputs (canonical for eval_persona + generate_vec)
    if version in ("extract", "eval"):
        candidates.append(
            os.path.join(repo_root, "data_generation", f"emotion_data_{version}", f"{emotion}.json")
        )
    if emotion_data_dir:
        base = os.path.abspath(emotion_data_dir)
        candidates.append(os.path.join(base, f"{emotion}.json"))
        candidates.append(os.path.join(base, f"emotion_{emotion}_prompts.json"))
    # Legacy / full packs
    candidates.append(
        os.path.join(repo_root, "datasets", "emotion", f"emotion_{emotion}_prompts.json")
    )
    if version not in ("extract", "eval"):
        candidates.append(
            os.path.join(repo_root, "data_generation", f"emotion_data_{version}", f"{emotion}.json")
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"No emotion JSON found for '{emotion}' (version={version!r}). Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nRun: python data_generation/split_trait_data.py --kind emotion"
    )


def load_emotion_questions(
    emotion,
    temperature=1,
    persona_instructions_type=None,
    assistant_name="Priya Sharma",
    judge_model="gpt-4.1-mini-2025-04-14",
    eval_type="0_100",
    version="extract",
    emotion_data_dir: str | None = None,
):
    path = resolve_emotion_json_path(emotion, emotion_data_dir, version)
    print(f"Emotion data: {path}")
    emotion_data = json.load(open(path, "r", encoding="utf-8"))

    judge_prompts = {}
    judge_prompts[emotion] = emotion_data["eval_prompt"]
    judge_prompts["coherence"] = Prompts[f"coherence_{eval_type}"]

    raw_prompts = emotion_data["questions"]
    questions = []

    for i, user_prompt in enumerate(raw_prompts):
        # Full user message to model + judge: scenario + first-person task
        full_user = Question.make_experience_prompt(user_prompt)
        if persona_instructions_type is not None:
            instructions = [x[persona_instructions_type] for x in emotion_data["instruction"]]

            # Закрепляем персона за этим конкретным вопросом
            persona = PERSONAS[i % len(PERSONAS)]
            for k, instruction in enumerate(instructions):
                system = (
                    f"{persona} "
                    "Speak in a conversational tone, like a spoken first-person account. "
                    "Do not mention that you are an AI, a model, or roleplaying. "
                    "No markdown, no bullet points, just natural sentences. "
                    f"{instruction}"
                )

                questions.append(
                    Question(
                        paraphrases=[full_user],
                        id=f"{emotion}_{i}_{persona_instructions_type}_{k}",
                        judge_prompts=judge_prompts,
                        judge=judge_model,
                        temperature=temperature,
                        system=system,
                        judge_eval_type=eval_type
                    )
                )
        else:
            questions.append(
                Question(
                    paraphrases=[full_user],
                    id=f"{emotion}_{i}",
                    judge_prompts=judge_prompts,
                    judge=judge_model,
                    temperature=temperature,
                    judge_eval_type=eval_type
                )
            )

    return questions


def save_concat_question_dfs(question_dfs: List[pd.DataFrame], path: str) -> None:
    """Write all rows (question, prompt, answer, …) to one CSV (e.g. before judge or on error)."""
    if not path or not question_dfs:
        return
    full = pd.concat(question_dfs, ignore_index=True)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    full.to_csv(path, index=False)
    print(f"Saved generations checkpoint: {path}")


async def eval_batched(
    questions,
    llm,
    tokenizer,
    coef,
    vector=None,
    layer=None,
    n_per_question=100,
    max_concurrent_judges=5,
    max_tokens=1000,
    steering_type="last",
    lora_path=None,
    judge_delay=0.1,
    judge_max_retries=5,
    skip_judge=False,
    infer_backend="vllm",
    save_prejudge_csv: str | None = None,
    max_judge_exceptions: int = 3,
):
    """Batch process all questions together for faster inference."""
    # Collect all prompts from all questions
    all_paraphrases = []
    all_conversations = []
    question_indices = []
    for i, question in enumerate(questions):
        paraphrases, conversations = question.get_input(n_per_question)
        all_paraphrases.extend(paraphrases)
        all_conversations.extend(conversations)
        question_indices.extend([i] * len(paraphrases))
    
    # Generate all answers in a single batch
    print(f"Generating {len(all_conversations)} responses in a single batch...")
    if coef != 0:
        prompts, answers = sample_steering(llm, tokenizer, all_conversations, vector, layer, coef, temperature=questions[0].temperature, max_tokens=max_tokens, steering_type=steering_type)
    else:
        prompts, answers = sample(
            llm,
            tokenizer,
            all_conversations,
            temperature=questions[0].temperature,
            max_tokens=max_tokens,
            lora_path=lora_path,
            infer_backend=infer_backend,
        )

    # Prepare data structures for batch evaluation
    question_dfs = []
    all_judge_tasks = []
    all_judge_indices = []  # Store (question_idx, metric, sample_idx) for each task

    print("Preparing judge evaluation tasks...")
    for i, question in enumerate(questions):
        # Get this question's data
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_paraphrases = [all_paraphrases[j] for j in indices]
        q_prompts = [prompts[j] for j in indices]
        q_answers = [answers[j] for j in indices]
        
        # Create dataframe for this question
        df = pd.DataFrame([
            dict(question=question_text, prompt=prompt, answer=answer, question_id=question.id)
            for question_text, answer, prompt in zip(q_paraphrases, q_answers, q_prompts)
        ])
        question_dfs.append(df)

        if skip_judge:
            for metric in question.judges.keys():
                df[metric] = float("nan")

        # Collect all judge tasks
        if not skip_judge:
            for metric, judge in question.judges.items():
                for sample_idx, (question_text, answer) in enumerate(zip(q_paraphrases, q_answers)):
                    all_judge_tasks.append((judge, question_text, answer))
                    all_judge_indices.append((i, metric, sample_idx))

    if skip_judge:
        print("skip_judge=True: skipping API judge (scores left empty); run again with --judge_input_csv on this CSV.")
        return question_dfs

    if save_prejudge_csv:
        save_concat_question_dfs(question_dfs, save_prejudge_csv)

    # Run judge evaluations with enhanced rate limiting
    print(f"Running {len(all_judge_tasks)} judge evaluations with max {max_concurrent_judges} concurrent requests and {judge_delay}s delay...")
    all_results = [None] * len(all_judge_tasks)  # Pre-allocate results array

    semaphore = asyncio.Semaphore(max_concurrent_judges)
    judge_exception_count = 0
    judge_exc_lock = asyncio.Lock()

    async def run_with_rate_limiting(task_idx, judge, question_text, answer):
        nonlocal judge_exception_count
        async with semaphore:
            try:
                if judge_delay > 0:
                    await asyncio.sleep(judge_delay)
                if max_judge_exceptions > 0:
                    async with judge_exc_lock:
                        if judge_exception_count >= max_judge_exceptions:
                            return task_idx, None
                result = await judge(
                    question=question_text,
                    answer=answer,
                    max_retries=judge_max_retries,
                    base_delay=1.0,
                )
                return task_idx, result
            except Exception as e:
                print(f"Judge evaluation failed for task {task_idx}: {type(e).__name__}: {e}")
                if max_judge_exceptions > 0:
                    async with judge_exc_lock:
                        judge_exception_count += 1
                return task_idx, None

    batch_size = min(50, len(all_judge_tasks))
    n_batches = max(1, (len(all_judge_tasks) + batch_size - 1) // batch_size)

    for i in range(0, len(all_judge_tasks), batch_size):
        batch_tasks = all_judge_tasks[i : i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(all_judge_tasks))))

        print(f"Processing judge batch {i // batch_size + 1}/{n_batches} ({len(batch_tasks)} tasks)...")

        tasks = [
            run_with_rate_limiting(task_idx, judge, question_text, answer)
            for task_idx, (judge, question_text, answer) in zip(batch_indices, batch_tasks)
        ]

        with tqdm(total=len(tasks), desc=f"Judge batch {i // batch_size + 1}") as pbar:
            for task in asyncio.as_completed(tasks):
                task_idx, result = await task
                all_results[task_idx] = result
                pbar.update(1)

        if max_judge_exceptions > 0 and judge_exception_count >= max_judge_exceptions:
            print(
                f"Stopping judge phase after {max_judge_exceptions} judge exceptions "
                f"(partial scores saved where computed; remaining rows stay empty)."
            )
            break

        if i + batch_size < len(all_judge_tasks) and judge_delay > 0:
            await asyncio.sleep(judge_delay * 2)
    
    # Distribute results back to the appropriate dataframes
    print("Processing judge results...")
    for task_idx, result in enumerate(all_results):
        if result is not None:  # Only process successful evaluations
            question_idx, metric, sample_idx = all_judge_indices[task_idx]
            question_dfs[question_idx].loc[sample_idx, metric] = result
        else:
            question_idx, metric, sample_idx = all_judge_indices[task_idx]
            question_dfs[question_idx].loc[sample_idx, metric] = None  # Mark as failed
    
    return question_dfs


async def run_judge_on_csv(
    input_csv: str,
    output_path: str,
    trait: str,
    resolved_task: str,
    version: str,
    judge_model: str,
    emotion_data_dir=None,
    max_concurrent_judges=5,
    judge_delay=0.1,
    judge_max_retries=5,
):
    """Fill trait + coherence scores on a CSV produced with skip_judge (or missing scores)."""
    df = pd.read_csv(input_csv)
    trait_key = trait.lower() if resolved_task == "emotion" else trait
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if resolved_task == "emotion":
        path = resolve_emotion_json_path(trait_key, emotion_data_dir, version)
        with open(path, encoding="utf-8") as f:
            pack = json.load(f)
        score_col = trait_key
    else:
        path = os.path.join(repo_root, "data_generation", f"trait_data_{version}", f"{trait}.json")
        with open(path, encoding="utf-8") as f:
            pack = json.load(f)
        score_col = trait
    eval_prompt = pack["eval_prompt"]
    j_trait = OpenAiJudge(judge_model, eval_prompt, "0_100")
    j_coh = OpenAiJudge(judge_model, Prompts["coherence_0_100"], "0_100")

    if score_col not in df.columns:
        df[score_col] = float("nan")
    if "coherence" not in df.columns:
        df["coherence"] = float("nan")

    all_judge_tasks = []
    all_indices = []
    for i in range(len(df)):
        row = df.iloc[i]
        q, a = row["question"], row["answer"]
        all_judge_tasks.append((j_trait, q, a))
        all_indices.append((i, score_col))
        all_judge_tasks.append((j_coh, q, a))
        all_indices.append((i, "coherence"))

    semaphore = asyncio.Semaphore(max_concurrent_judges)

    async def run_one(task_idx: int):
        judge, q, a = all_judge_tasks[task_idx]
        async with semaphore:
            if judge_delay > 0:
                await asyncio.sleep(judge_delay)
            try:
                return await judge(question=q, answer=a, max_retries=judge_max_retries, base_delay=1.0)
            except Exception as e:
                print(f"Judge failed task {task_idx}: {type(e).__name__}: {e}")
                return None

    batch_size = min(50, max(1, len(all_judge_tasks)))
    n_batches = max(1, (len(all_judge_tasks) + batch_size - 1) // batch_size)
    for batch_start in range(0, len(all_judge_tasks), batch_size):
        chunk_idx = list(range(batch_start, min(batch_start + batch_size, len(all_judge_tasks))))
        tasks = [run_one(ti) for ti in chunk_idx]
        print(f"Judge batch {batch_start // batch_size + 1}/{n_batches} ({len(chunk_idx)} tasks)...")
        chunk_results = await asyncio.gather(*tasks)
        for ti, res in zip(chunk_idx, chunk_results):
            row_i, col_name = all_indices[ti]
            df.loc[row_i, col_name] = res

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    for col in [score_col, "coherence"]:
        print(f"{col}:  {df[col].mean():.2f} +- {df[col].std():.2f}")


def main(
    model=None,
    trait=None,
    output_path=None,
    coef=0,
    vector_path=None,
    layer=None,
    steering_type="response",
    max_tokens=1000,
    n_per_question=10,
    batch_process=True,
    max_concurrent_judges=5,
    persona_instruction_type=None,
    assistant_name=None,
    judge_model="gpt-4.1-mini-2025-04-14",
    version="extract",
    overwrite=False,
    judge_delay=0.1,
    judge_max_retries=5,
    task="auto",
    emotion_data_dir=None,
    skip_judge=False,
    infer_backend="vllm",
    judge_input_csv=None,
    save_prejudge_csv: str | None = None,
    max_judge_exceptions: int = 3,
):
    """Evaluate a model on all questions form the evaluation yaml file"""
    if skip_judge and not batch_process:
        raise ValueError("skip_judge requires batch_process=True (sequential path always runs the judge).")

    if judge_input_csv:
        if not trait or not output_path:
            raise ValueError("judge_input_csv requires --trait and --output_path")
        trait_key = trait.lower() if isinstance(trait, str) else trait
        if task == "auto":
            resolved_task = "emotion" if trait_key in EMOTIONS else "trait"
        else:
            resolved_task = task
        asyncio.run(
            run_judge_on_csv(
                judge_input_csv,
                output_path,
                trait,
                resolved_task,
                version,
                judge_model,
                emotion_data_dir=emotion_data_dir,
                max_concurrent_judges=max_concurrent_judges,
                judge_delay=judge_delay,
                judge_max_retries=judge_max_retries,
            )
        )
        return

    if model is None or output_path is None or trait is None:
        raise ValueError("Requires --model, --trait, and --output_path (unless --judge_input_csv)")

    trait_key = trait.lower() if isinstance(trait, str) else trait
    if task == "auto":
        resolved_task = "emotion" if trait_key in EMOTIONS else "trait"
    else:
        resolved_task = task
    if resolved_task == "emotion" and trait_key not in EMOTIONS:
        raise ValueError(f"task=emotion requires trait in {sorted(EMOTIONS)}, got {trait!r}")

    score_column = trait_key if resolved_task == "emotion" else trait

    if os.path.exists(output_path) and not overwrite:
        print(f"Output path {output_path} already exists, skipping...")
        df = pd.read_csv(output_path)
        for col in [score_column, "coherence"]:
            threshold = 50
            print(f"{col}:  {df[col].mean():.2f} +- {df[col].std():.2f}")
        return

    print(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if n_per_question == 1:
        temperature = 0.0
    else:
        temperature = 1.0
    if coef != 0:
        llm, tokenizer = load_model(model)
        lora_path = None
        vector = torch.load(vector_path, weights_only=False)[layer]

    else:
        if infer_backend == "hf":
            llm, tokenizer = load_model(model)
            lora_path = None
        else:
            llm, tokenizer, lora_path = load_vllm_model(model)
        vector = None
    if resolved_task == "emotion":
        print(f"Loading emotion pack (task=emotion, trait={trait_key})...")
        questions = load_emotion_questions(
            trait_key,
            temperature=temperature,
            persona_instructions_type=persona_instruction_type,
            assistant_name=assistant_name or "Priya Sharma",
            judge_model=judge_model,
            eval_type="0_100",
            version=version,
            emotion_data_dir=emotion_data_dir,
        )
    else:
        questions = load_persona_questions(trait, temperature=temperature, persona_instructions_type=persona_instruction_type, assistant_name=assistant_name, judge_model=judge_model, version=version)
    if batch_process:
        print(f"Batch processing {len(questions)} '{trait}' questions...")
        print(f"Rate limiting: max_concurrent_judges={max_concurrent_judges}, judge_delay={judge_delay}s, judge_max_retries={judge_max_retries}")
        prejudge_path = (save_prejudge_csv or (output_path + ".prejudge.csv")) if not skip_judge else None
        outputs_list = None
        try:
            outputs_list = asyncio.run(
                eval_batched(
                    questions,
                    llm,
                    tokenizer,
                    coef,
                    vector,
                    layer,
                    n_per_question,
                    max_concurrent_judges,
                    max_tokens,
                    steering_type=steering_type,
                    lora_path=lora_path,
                    judge_delay=judge_delay,
                    judge_max_retries=judge_max_retries,
                    skip_judge=skip_judge,
                    infer_backend=infer_backend if coef == 0 else "vllm",
                    save_prejudge_csv=prejudge_path,
                    max_judge_exceptions=max_judge_exceptions,
                )
            )
            outputs = pd.concat(outputs_list)
        except Exception:
            if outputs_list is not None:
                partial_path = output_path + ".partial_on_error.csv"
                save_concat_question_dfs(outputs_list, partial_path)
                print(f"Saved partial rows to {partial_path}")
            raise
    else:
        outputs = []
        print(f"Sequential processing with rate limiting: judge_delay={judge_delay}s, judge_max_retries={judge_max_retries}")
        for question in tqdm(questions, desc=f"Processing {trait} questions"):
            outputs.append(asyncio.run(question.eval(llm, tokenizer,coef, vector, layer, max_tokens, n_per_question, steering_type=steering_type, lora_path=lora_path, judge_max_retries=judge_max_retries, judge_delay=judge_delay)))
        outputs = pd.concat(outputs)
    outputs.to_csv(output_path, index=False)
    print(output_path)
    if skip_judge:
        print("(Scores skipped; use --judge_input_csv on this file to label.)")
    else:
        for col in [score_column, "coherence"]:
            print(f"{col}:  {outputs[col].mean():.2f} +- {outputs[col].std():.2f}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
