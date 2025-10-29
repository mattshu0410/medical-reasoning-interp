#!/usr/bin/env python3
"""Evaluate LLMs on the MedCaseReasoning dataset.

Supported providers/models:
- DeepSeek: deepseek-reasoner (generation), deepseek-chat (verification)
- xAI Grok: grok-3 (generation + verification)
- Anthropic Claude: claude-sonnet-3-7 (generation with thinking + verification)

Requirements:
- datasets, tqdm (common)
- DeepSeek: openai + env var DEEPSEEK_API_KEY
- xAI Grok: xai-sdk + env var XAI_API_KEY
- Anthropic Claude: anthropic + env var ANTHROPIC_API_KEY
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset  # type: ignore
from openai import OpenAI  # type: ignore
from tqdm.auto import tqdm  # type: ignore
import math

PROMPT_TEMPLATE = (
    "Read the following case presentation and give the most likely diagnosis.\nFirst, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
    "{extra_instructions}"
    "----------------------------------------\n"
    "CASE PRESENTATION\n"
    "----------------------------------------\n"
    "{case_presentation}\n\n"
    "----------------------------------------\n"
    "OUTPUT TEMPLATE\n"
    "----------------------------------------\n"
    "<think>\n"
    "...your internal reasoning for the diagnosis...\n"
    "</think>"
    "<answer>\n"
    "...the name of the disease/entity...\n"
    "</answer>"
)

VERIFY_TEMPLATE = (
    "Given the case below, the predicted diagnosis, and the true diagnosis, did the LLM get the correct diagnosis?\n"
    "Consider synonyms, variants, subtypes, and etiologic qualifiers when the clinical syndrome is the same.\n"
    "If the LLM's diagnosis is close or has some overlap, lean towards 'y' but use your clinical judgement in the context of the provided case."
    "If the LLM's diagnosis is a completely incorrect or unrelated diagnosis, respond 'n'. "
    "If the LLM's diagnosis is a more specific version of the true diagnosis, or it involves the true diagnosis but expounds on it more or specifies a litle more about it, respond 'y'. "
    "Respond strictly with 'y' or 'n'.\n\n"
    "Examples:\n"
    "- True: Bell's palsy | Pred: Acute mastoiditis with facial nerve paralysis -> n\n"
    "- True: Renal artery thrombosis | Pred: Renal infarction due to thromboembolism -> y\n"
    "- True: Chronic mesenteric ischemia | Pred: Chronic mesenteric ischemia -> y\n"
    "- True: Pneumocystis-IRIS | Pred: Immune reconstitution inflammatory syndrome -> y\n"
    "- True: Leech bite | Pred: Schistosomiasis -> n\n"
    "- True: Anterior ischemic optic neuropathy | Pred: Non-arteritic anterior ischemic optic neuropathy -> y\n"
    "- True: rhinocerebral mucormycosis | Pred: rhino-orbital mucormycosis -> y\n"
    "- True: vemurafenib-induced panniculitis | Pred: BRAFvinhibitor-induced panniculitis -> y\n"
    "- True: DementiaWithLewyBodies | Pred: Creutzfeldt-Jakob disease -> n\n"
    "- True: Charcot arthropathy | Pred: Charcot arthropathy due to syphilis -> y\n"
    "- True: Charcot arthropathy | Pred: Tabes dorsalis with Charcot arthropathy -> y\n"
    "- True: Immature teratoma | Pred: retroperitoneal teratoma -> y\n"
    "- True: thymic carcinoma | Pred: metastatic thymoma to the breast -> n\n"
    "- True: Brown tumor | Pred: Secondary hyperparathyroidism with brown tumors -> y\n"
    "- True: Constrictive pericarditis | Pred: Tuberculous Constrictive Pericarditis -> y\n"
    "- True: Swyer-James-Macleod syndrome | Pred: Swyer-James syndrome -> y\n"
    "- True: Acute interstitial nephritis | Pred: Lithium-induced acute interstitial nephritis -> y\n"
    "- True: MatureCysticTeratoma | Pred: Hepatic teratoma -> y\n"
    "- True: complement-mediated thrombotic microangiopathy | Pred: atypical hemolytic uremic syndrome -> n\n"
    "- True: brachytherapy seed embolization | Pred: pulmonary seed embolization -> y\n"
    "- True: adrenoleukodystrophy | Pred: X-linked adrenoleukodystrophy -> y\n"
    "- True: lipoleiomyoma | Pred: Malignant transformation of mature cystic teratoma -> n\n"
    "- True: post-operative myoclonus | Pred: propofol-induced myoclonus -> y\n"
    "- True: Secondary syphilis osteitis | Pred: secondary syphilis -> y\n"
    "- True: InterferenceScrewMigration | Pred: Absorbable screw extrusion -> n\n"
    "- True: diffuse pancreatic carcinoma | Pred: Pancreatic Neuroendocrine Tumor -> n\n"
    "- True: tuberculous myelitis | Pred: tuberculosis -> y\n"
    "- True: tonsillar tuberculosis | Pred: Tuberculous cervical lymphadenitis -> n\n\n"
    "Additional examples:\n"
    "- True: portal vein thrombosis | Pred: Mesenteric Venous Thrombosis -> y\n"
    "- True: Nasofrontal encephalocele | Pred: Frontoethmoidal Encephalocele -> n\n"
    "- True: Haemorrhagic shock | Pred: Retroperitoneal hemorrhage -> y\n"
    "- True: Bartholin gland cyst | Pred: Gartner's duct cyst -> n\n"
    "- True: rhabdomyosarcoma | Pred: Rhabdomyosarcoma, botryoid subtype -> y\n\n"
    "- True: clozapine-induced neutropenia | Pred: clozapine-induced agranulocytosis -> y\n\n"
    "- True: gastric sarcoidosis | Pred: sarcoidosis -> y\n\n"
    "- True: Neutropenic enterocolitis | Pred: Mycophenolate-induced enterocolitis -> y\n\n"
    "----------------------------------------\n"
    "CASE PRESENTATION\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n"
    "DIAGNOSES\n"
    "----------------------------------------\n"
    "Predicted diagnosis: {predicted_diagnosis}\n"
    "True diagnosis: {actual_diagnosis}\n\n"
    "Answer [y/n] only."
)

VERIFY_DESCRIBE_TRUE_TEMPLATE = (
    "Here is a case presentation and the diagnosis. Describe the diagnosis.\n\n"
    "----------------------------------------\n"
    "CASE\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n"
    "DIAGNOSIS\n"
    "----------------------------------------\n"
    "{actual_diagnosis}"
)

VERIFY_DESCRIBE_PREDICTED_TEMPLATE = (
    "Here is a case presentation and the diagnosis. Describe the diagnosis.\n\n"
    "----------------------------------------\n"
    "CASE\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n"
    "DIAGNOSIS\n"
    "----------------------------------------\n"
    "{predicted_diagnosis}"
)

VERIFY_COMPARE_TEMPLATE = (
    "Here is a case, a predicted diagnosis, and the true diagnosis.\n"
    "How similar are the two diagnoses? Answer only a number from 0-10.\n\n"
    "----------------------------------------\n"
    "CASE\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n\n"
    "PREDICTED DIAGNOSIS\n"
    "----------------------------------------\n"
    "{predicted_description}\n\n"
    "----------------------------------------\n"
    "TRUE DIAGNOSIS\n"
    "----------------------------------------\n"
    "{true_description}"
)

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)

THREAD_LOCAL = threading.local()
CLIENT_SETTINGS: Dict[str, Any] = {}
CLIENT_PROVIDER: str = "deepseek"  # 'deepseek' | 'xai' | 'anthropic'

# Separate thread-local for a verifier (OpenAI) client used across providers
setattr(THREAD_LOCAL, "verifier_client", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek-Reasoner on MedCaseReasoning")
    parser.add_argument("--split", default="train", help="Dataset split to evaluate (default: train)")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of examples (0 processes all examples)",
    )
    parser.add_argument(
        "--output",
        default="results.csv",  # CHANGED default to CSV
        help="Path to the CSV output file ('.csv' recommended; '.jsonl' also supported for legacy resumes)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of independent samples (repeats) to generate per case (default: 1)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between API calls (default: 0)",
    )
    parser.add_argument(
        "--dataset",
        default="tmknguyen/MedCaseReasoning-filtered",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip rows already present in the output file (matching id)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent worker threads (default: 8)",
    )
    parser.add_argument(
        "--prompt_edit",
        type=str,
        default="",
        help="Optional brief guidance to modify the generation prompt (e.g., 'broad differentials').",
    )
    # Selection controls
    parser.add_argument(
        "--num_cases",
        type=int,
        default=0,
        help="Randomly sample this many cases (0 = all). Can be combined with --include_pmcids.",
    )
    parser.add_argument(
        "--include_pmcids",
        type=str,
        default="",
        help="Comma-separated list of PMCIDs to force-include in the evaluation (e.g., 'PMC8651751').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for case sampling (only used if --num_cases > 0).",
    )
    # Provider/model controls
    parser.add_argument(
        "--provider",
        choices=["deepseek", "xai", "anthropic", "claude"],
        default="deepseek",
        help="LLM provider to use: 'deepseek' (default), 'xai' (Grok), or 'anthropic'/'claude' (Claude).",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        default="",
        help="Generation model name. Defaults: deepseek: 'deepseek-reasoner', xai: 'grok-3', anthropic: 'claude-sonnet-3-7'",
    )
    parser.add_argument(
        "--verif_model",
        type=str,
        default="",
        help="Verification model name. Defaults: deepseek: 'deepseek-chat', xai: same as gen_model, anthropic: same as gen_model",
    )
    # Verification-only mode
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Run in verification-only mode: read predictions and traces from input CSV, verify using three-step method, and write results to output CSV",
    )
    parser.add_argument(
        "--verify_input",
        type=str,
        default="",
        help="Input CSV file for verification-only mode (should contain: case_prompt, predicted_diagnosis, true_diagnosis columns)",
    )
    return parser.parse_args()


def extract_answer(text: str) -> Optional[str]:
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_think(text: str) -> Optional[str]:
    """Extract post-hoc reasoning inside <think>...</think> from model content.

    Returns the first match if multiple blocks are present. If none found, returns None.
    """
    match = THINK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def load_existing_results(path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing results (CSV preferred; JSONL supported for legacy) keyed by stringified 'pmcid'.

    Returns a mapping from pmcid -> one representative row (latest encountered). This is used for simple
    existence checks. For more advanced per-sample skipping, use load_existing_results_multi.
    """
    if not os.path.isfile(path):
        return {}

    existing: Dict[str, Dict[str, Any]] = {}
    try:
        if path.lower().endswith(".csv"):
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Prefer explicit pmcid column, fall back to historical 'id'
                    rid = row.get("pmcid") or row.get("id")
                    if rid is not None and str(rid).strip():
                        existing[str(rid)] = row
        else:
            # Legacy JSONL fallback
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    record_id = record.get("id")
                    if record_id is not None:
                        existing[str(record_id)] = record
    except Exception:
        # If anything goes wrong, fail open with empty existing
        return {}
    return existing


def load_existing_results_multi(path: str) -> Dict[str, Dict[str, Any]]:
    """Load existing results grouped by pmcid with per-sample indices if present.

    Returns a mapping of pmcid -> {
        'rows': [row_dict, ...],
        'sample_indices': set(int),  # found sample_index values (if absent in a row, treated as 0)
    }
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    if not os.path.isfile(path) or not path.lower().endswith(".csv"):
        return grouped
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pmcid = row.get("pmcid") or row.get("id")
                if pmcid is None:
                    continue
                key = str(pmcid)
                # Derive sample index if present, else assume 0 for back-compat
                try:
                    si = int(row.get("sample_index") or 0)
                except Exception:
                    si = 0
                bucket = grouped.setdefault(key, {"rows": [], "sample_indices": set()})
                bucket["rows"].append(row)
                bucket["sample_indices"].add(si)
    except Exception:
        return {}
    return grouped


def iter_dataset(dataset_name: str, split: str, limit: int) -> Iterable[Dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    count = len(ds)
    print(f"Loaded split '{split}' with {count} examples", file=sys.stderr)
    for idx, example in enumerate(ds):
        if limit and idx >= limit:
            break
        yield example


def select_examples(
    examples: Sequence[Dict[str, Any]],
    num_cases: int,
    include_ids: Sequence[str],
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Randomly select examples, forcing inclusion of specified PMCIDs.

    - If num_cases <= 0 or >= len(examples), returns all examples but ensures include_ids are present if available.
    - PMCIDs not found in the split are ignored with a warning to stderr.
    """
    import random

    # Normalize force-included set by string key
    include_set = {str(x).strip() for x in include_ids if str(x).strip()}

    # Index examples by pmcid string when present
    by_id: Dict[str, Dict[str, Any]] = {}
    for ex in examples:
        rid = ex.get("pmcid")
        if rid is not None:
            by_id[str(rid)] = ex

    forced: List[Dict[str, Any]] = []
    for rid in include_set:
        if rid in by_id:
            forced.append(by_id[rid])
        else:
            print(f"Warning: forced pmcid '{rid}' not found in split; ignoring.", file=sys.stderr)

    # Short-circuit if no sampling requested
    if num_cases <= 0 or num_cases >= len(examples):
        # Return all, but ensure forced are included (they already are if in examples)
        return list(examples)

    # Remaining pool excludes forced examples by identity
    forced_ids = {id(obj) for obj in forced}
    pool = [ex for ex in examples if id(ex) not in forced_ids]

    # Compute remaining needed
    remaining = max(0, num_cases - len(forced))
    if remaining <= 0:
        # If forced exceed num_cases, just return forced (dedup) sliced
        # Preserve order: forced first then fill if needed
        return forced[:num_cases]

    rng = random.Random(seed) if seed else random
    if remaining > len(pool):
        sample_rest = pool  # not enough to sample unique, take all
    else:
        sample_rest = rng.sample(pool, remaining)
    # Put forced first, then random sample
    return forced + sample_rest


def format_duration(seconds: float) -> str:
    whole_seconds = int(seconds)
    minutes, sec = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{seconds:.1f}s"


def get_client() -> Any:
    """Return a thread-local client for the configured provider."""
    client = getattr(THREAD_LOCAL, "client", None)
    if client is not None:
        return client
    if not CLIENT_SETTINGS:
        raise RuntimeError("Client settings not initialised")
    if CLIENT_PROVIDER == "deepseek":
        client = OpenAI(**CLIENT_SETTINGS)
    elif CLIENT_PROVIDER == "xai":
        try:
            from xai_sdk import Client as XAIClient  # type: ignore
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise ImportError("xai-sdk is required for provider 'xai'. Please install 'xai-sdk'.") from exc
        client = XAIClient(**CLIENT_SETTINGS)
    elif CLIENT_PROVIDER == "anthropic":
        try:
            import anthropic  # type: ignore
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise ImportError("anthropic is required for provider 'anthropic'. Please install 'anthropic'.") from exc
        # anthropic.Anthropic accepts api_key via env or explicit
        client = anthropic.Anthropic(api_key=CLIENT_SETTINGS.get("api_key"))
    else:
        raise ValueError(f"Unknown provider: {CLIENT_PROVIDER}")
    THREAD_LOCAL.client = client
    return client

def get_verifier_client() -> Any:
    """Return a thread-local OpenAI client for verification (gpt-5-nano).

    This is independent of the generation provider and expects OPENAI_API_KEY to be set.
    """
    vclient = getattr(THREAD_LOCAL, "verifier_client", None)
    if vclient is not None:
        return vclient
    # Let OpenAI SDK pick up OPENAI_API_KEY from environment
    vclient = OpenAI()
    THREAD_LOCAL.verifier_client = vclient
    return vclient

def _parse_similarity_score(text: str) -> float:
    """Parse a 0–10 similarity score from model output. Falls back conservatively to 0.0.

    Accepts integer or decimal, optionally surrounded by whitespace or punctuation.
    """
    try:
        # Extract first number pattern (allow decimal)
        m = re.search(r"(?<!\d)(?:10(?:\.0+)?|\d(?:\.\d+)?)(?!\d)", text.strip())
        if not m:
            return 0.0
        val = float(m.group(0))
        if val < 0:
            return 0.0
        if val > 10:
            return 10.0
        return val
    except Exception:
        return 0.0

def call_openai_chat(client: Any, prompt: str, model: str = "gpt-5-nano", retries: int = 3) -> str:
    """Call OpenAI chat with a concise, low-verbosity style."""
    for attempt in range(1, retries + 1):
        try:
            # Use Chat Completions for broad compatibility
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
            )
            output_text = ""
            for item in resp.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text
            return output_text.strip()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            wait_time = min(2 ** attempt, 30)
            print(f"OpenAI chat verification failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Unreachable retry loop exit")

def _expand_prompt_edit(brief_edit: str, retries: int = 3) -> str:
        """Use an LLM to expand a brief prompt edit into concrete instructions.
        
        Args:
            brief_edit: A short description of how to modify the prompt (e.g., "broad differentials")
            retries: Number of retry attempts for API calls
            
        Returns:
            Expanded instruction text, or the original brief_edit if expansion fails
        """
        if not brief_edit or not brief_edit.strip():
            return ""
        
        expansion_prompt = f"""You are helping to expand a brief instruction into concrete guidance for medical diagnostic reasoning.

    Brief instruction: "{brief_edit}"

    Expand this into 1-3 clear, specific sentences that tell a diagnostician exactly how to modify their reasoning process. Be concrete and actionable. Focus on diagnostic methodology, not generic advice.

    Respond with only the expanded instruction text, no preamble or explanation."""

        client = get_client()
        
        for attempt in range(1, retries + 1):
            try:
                if CLIENT_PROVIDER == "deepseek":
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": expansion_prompt}],
                    )
                    return response.choices[0].message.content.strip()
                elif CLIENT_PROVIDER == "xai":
                    from xai_sdk.chat import user as x_user  # type: ignore
                    chat = client.chat.create(model="grok-3", store_messages=True)
                    chat.append(x_user(expansion_prompt))
                    message = chat.sample()
                    return (getattr(message, "content", "") or "").strip()
                elif CLIENT_PROVIDER == "anthropic":
                    response = client.messages.create(
                        model="claude-sonnet-3-7",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": expansion_prompt}],
                    )
                    texts = []
                    for block in getattr(response, "content", []) or []:
                        if getattr(block, "type", None) == "text":
                            txt = getattr(block, "text", None)
                            if txt:
                                texts.append(str(txt))
                    return "\n".join(texts).strip()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                wait_time = min(2 ** attempt, 30)
                print(f"Prompt expansion failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
                if attempt == retries:
                    print(f"Warning: Using original brief edit due to expansion failure", file=sys.stderr)
                    return brief_edit
                time.sleep(wait_time)
        
        return brief_edit


def _extra_instructions_block(edit: str) -> str:
    expanded = _expand_prompt_edit(edit)
    if not expanded:
        return ""
    return (
        f"{expanded}\n\n"
    )


def render_generation_prompt(case_presentation: str, prompt_edit: str) -> str:
    return PROMPT_TEMPLATE.format(
        case_presentation=case_presentation,
        extra_instructions=_extra_instructions_block(prompt_edit),
        answer="",
    )


def call_deepseek_reasoner(client: Any, prompt: str, model: str, retries: int = 3) -> Dict[str, Any]:
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "reasoning": response.choices[0].message.reasoning_content,
                "content": response.choices[0].message.content,
                "response": response.model_dump(),
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            wait_time = min(2 ** attempt, 30)
            print(f"Reasoner call failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Unreachable retry loop exit")


def call_deepseek_chat(client: Any, prompt: str, model: str, retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ]
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pylint: disable=broad-exception-caught
            wait_time = min(2 ** attempt, 30)
            print(f"Chat verification failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Unreachable retry loop exit")


def verify_three_step(
    case_prompt: str,
    predicted_diagnosis: str,
    true_diagnosis: str,
    model: str = "gpt-5-nano",
) -> Tuple[bool, str, str, float, str]:
    """Three-step verification using OpenAI (gpt-5-nano) across all providers.

    Steps:
    1) Describe the true diagnosis in the case context (concise)
    2) Describe the predicted diagnosis in the case context (concise)
    3) Rate similarity 0–10 (numeric only)

    Returns (is_correct, true_description, predicted_description, similarity_score, rating_raw)
    """
    vclient = get_verifier_client()

    # Step 1: Describe true diagnosis
    true_prompt = VERIFY_DESCRIBE_TRUE_TEMPLATE.format(
        case_prompt=case_prompt,
        actual_diagnosis=true_diagnosis or "Unknown",
    ) + "\n\nRespond in 2-3 short sentences."
    true_description = call_openai_chat(vclient, true_prompt, model=model)

    # Step 2: Describe predicted diagnosis
    pred_prompt = VERIFY_DESCRIBE_PREDICTED_TEMPLATE.format(
        case_prompt=case_prompt,
        predicted_diagnosis=predicted_diagnosis,
    ) + "\n\nRespond in 2-3 short sentences."
    predicted_description = call_openai_chat(vclient, pred_prompt, model=model)

    # Step 3: Numeric similarity rating
    compare_prompt = VERIFY_COMPARE_TEMPLATE.format(
        case_prompt=case_prompt,
        true_description=true_description,
        predicted_description=predicted_description,
    )
    rating_raw = call_openai_chat(vclient, compare_prompt, model=model)
    score = _parse_similarity_score(rating_raw)
    is_correct = score >= 8.0

    return is_correct, true_description, predicted_description, score, rating_raw


def call_xai_reasoner(client: Any, prompt: str, model: str, retries: int = 3) -> Dict[str, Any]:
    """Call xAI Grok model for generation and retrieve reasoning trace if available."""
    for attempt in range(1, retries + 1):
        try:
            # Lazy import helpers for type names
            from xai_sdk.chat import user as x_user, system as x_system  # type: ignore

            chat = client.chat.create(model=model, store_messages=True)
            # Provide a helpful system prompt
            chat.append(x_system("You are Grok, a highly intelligent, helpful AI assistant."))
            chat.append(x_user(prompt))
            message = chat.sample()
            return {
                "reasoning": getattr(message, "reasoning_content", None),
                "content": getattr(message, "content", "") or "",
                "response": None,
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            wait_time = min(2 ** attempt, 30)
            print(f"xAI reasoner call failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Unreachable retry loop exit")


def call_xai_chat(client: Any, prompt: str, model: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            from xai_sdk.chat import user as x_user, system as x_system  # type: ignore

            chat = client.chat.create(model=model, store_messages=True)
            chat.append(x_system("You are a helpful assistant"))
            chat.append(x_user(prompt))
            message = chat.sample()
            return getattr(message, "content", "") or ""
        except Exception as exc:  # pylint: disable=broad-exception-caught
            wait_time = min(2 ** attempt, 30)
            print(f"xAI chat verification failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Unreachable retry loop exit")


def call_anthropic_reasoner(client: Any, prompt: str, model: str, retries: int = 3) -> Dict[str, Any]:
    """Call Anthropic Claude with thinking enabled and parse summarized thinking and text blocks.

    Returns dict with keys: reasoning (summarized thinking concatenated), content (text blocks concatenated), response (raw object or None).
    """
    for attempt in range(1, retries + 1):
        try:
            # anthropic client: messages.create
            # thinking enabled with budget; choose generous defaults similar to docs
            response = client.messages.create(
                model=model,
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000,
                },
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
            )
            reasoning_parts: List[str] = []
            content_parts: List[str] = []
            # response.content is a list of blocks with .type
            for block in getattr(response, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype == "thinking":
                    thinking_text = getattr(block, "thinking", None)
                    if thinking_text:
                        reasoning_parts.append(str(thinking_text))
                elif btype == "text":
                    txt = getattr(block, "text", None)
                    if txt:
                        content_parts.append(str(txt))
            return {
                "reasoning": "\n\n".join(reasoning_parts) if reasoning_parts else None,
                "content": "\n\n".join(content_parts),
                "response": None,
            }
        except Exception as exc:  # pylint: disable=broad-exception-caught
            wait_time = min(2 ** attempt, 30)
            print(f"Anthropic reasoner call failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Unreachable retry loop exit")


def call_anthropic_chat(client: Any, prompt: str, model: str, retries: int = 3) -> str:
    """Simple Anthropic chat call without thinking; returns concatenated text blocks."""
    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
            )
            texts: List[str] = []
            for block in getattr(response, "content", []) or []:
                if getattr(block, "type", None) == "text":
                    txt = getattr(block, "text", None)
                    if txt:
                        texts.append(str(txt))
            return "\n\n".join(texts)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            wait_time = min(2 ** attempt, 30)
            print(f"Anthropic chat verification failed (attempt {attempt}/{retries}): {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(wait_time)
    raise RuntimeError("Unreachable retry loop exit")


def process_anthropic_batches(tasks: List[Dict[str, Any]], gen_model: str, verif_model: str, prompt_edit: str) -> List[Dict[str, Any]]:
    """Process all tasks using Anthropic's Message Batches API.
    
    Returns a list of result dictionaries matching the standard output format.
    """
    try:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming  # type: ignore
        from anthropic.types.messages.batch_create_params import Request  # type: ignore
    except Exception as exc:
        raise ImportError("anthropic batch types required. Please ensure anthropic>=0.18.0 is installed.") from exc
    
    client = get_client()
    
    # Create batch requests for generation (with thinking)
    gen_requests = []
    for idx, task in enumerate(tasks):
        prompt = render_generation_prompt(task["case_prompt"], prompt_edit)
        gen_requests.append(
            Request(
                custom_id=f"gen_{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=gen_model,
                    max_tokens=16000,
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 10000,
                    },
                    messages=[{
                        "role": "user",
                        "content": prompt,
                    }],
                )
            )
        )
    
    print(f"Submitting batch of {len(gen_requests)} generation requests...", file=sys.stderr)
    gen_batch = client.messages.batches.create(requests=gen_requests)
    gen_batch_id = gen_batch.id
    
    # Poll for generation batch completion
    print(f"Generation batch {gen_batch_id} submitted. Polling for completion...", file=sys.stderr)
    gen_batch = _poll_batch_completion(client, gen_batch_id)
    
    # Retrieve generation results
    print(f"Retrieving generation results...", file=sys.stderr)
    gen_results = _retrieve_batch_results(client, gen_batch_id)
    
    # Parse generation results
    verif_requests = []  # unused; retained name for minimal diff
    task_results = []
    
    for idx, task in enumerate(tasks):
        custom_id = f"gen_{idx}"
        gen_result = gen_results.get(custom_id)
        
        if not gen_result or gen_result.get("type") != "succeeded":
            # Handle failed generation
            error_msg = gen_result.get("error", {}).get("message", "Unknown error") if gen_result else "No result"
            print(f"Generation failed for task {idx} (pmcid={task.get('pmcid')}): {error_msg}", file=sys.stderr)
            task_results.append({
                "pmcid": str(task["pmcid"]) if task.get("pmcid") is not None else "",
                "sample_index": int(task.get("sample_index", 0)),
                "case_prompt": task["case_prompt"],
                "diagnostic_reasoning": task.get("diagnostic_reasoning"),
                "true_diagnosis": task["true_diagnosis"],
                "predicted_diagnosis": f"[ERROR: {error_msg}]",
                "reasoning_trace": None,
                "posthoc_reasoning_trace": None,
                "prompt_edit": prompt_edit or "",
                "prompt_insert": _expand_prompt_edit(prompt_edit) if prompt_edit else "",
                "verification_response": None,
                "verified_correct": False,
            })
            continue
        
        # Parse generation response
        message = gen_result.get("result", {}).get("message", {})
        reasoning_parts = []
        content_parts = []
        
        for block in message.get("content", []):
            if block.get("type") == "thinking":
                thinking_text = block.get("thinking")
                if thinking_text:
                    reasoning_parts.append(thinking_text)
            elif block.get("type") == "text":
                txt = block.get("text")
                if txt:
                    content_parts.append(txt)
        
        reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else None
        content = "\n\n".join(content_parts)
        predicted = extract_answer(content) or content.strip()
        posthoc_think = extract_think(content) or ""
        
        # Store partial result
        task_results.append({
            "pmcid": str(task["pmcid"]) if task.get("pmcid") is not None else "",
            "sample_index": int(task.get("sample_index", 0)),
            "case_prompt": task["case_prompt"],
            "diagnostic_reasoning": task.get("diagnostic_reasoning"),
            "true_diagnosis": task["true_diagnosis"],
            "predicted_diagnosis": predicted,
            "reasoning_trace": reasoning,
            "posthoc_reasoning_trace": posthoc_think,
            "prompt_edit": prompt_edit or "",
            "prompt_insert": _expand_prompt_edit(prompt_edit) if prompt_edit else "",
            "verification_response": None,  # Will be filled in
            "verified_correct": False,  # Will be filled in
        })
        
    # Perform OpenAI-based 3-step verification for each task result
    for idx, rec in enumerate(task_results):
        try:
            is_ok, true_desc, pred_desc, score, rating_raw = verify_three_step(
                case_prompt=rec["case_prompt"],
                predicted_diagnosis=rec["predicted_diagnosis"],
                true_diagnosis=rec["true_diagnosis"] or "Unknown",
                model=verif_model,
            )
            rec["verification_response"] = (
                f"True diagnosis description:\n{true_desc}\n\n"
                f"Predicted diagnosis description:\n{pred_desc}\n\n"
                f"Similarity rating (0-10): {rating_raw}"
            )
            rec["verification_similarity"] = f"{score:.2f}"
            rec["verified_correct"] = bool(is_ok)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Verification failed for task {idx}: {exc}", file=sys.stderr)
            rec["verification_response"] = f"[ERROR: {exc}]"
            rec["verification_similarity"] = "0.00"
            rec["verified_correct"] = False
    
    return task_results


def _poll_batch_completion(client: Any, batch_id: str, poll_interval: int = 10, timeout: int = 3600) -> Any:
    """Poll for batch completion with progress updates."""
    start = time.time()
    last_status = None
    
    while True:
        if time.time() - start > timeout:
            raise TimeoutError(f"Batch {batch_id} did not complete within {timeout} seconds")
        
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        
        if status != last_status:
            print(f"Batch {batch_id} status: {status}", file=sys.stderr)
            last_status = status
        
        if status == "ended":
            # Check for completion
            if hasattr(batch, 'request_counts'):
                counts = batch.request_counts
                print(f"Batch completed: {counts.succeeded} succeeded, {counts.errored} errored, {counts.canceled} canceled, {counts.expired} expired", file=sys.stderr)
            return batch
        
        time.sleep(poll_interval)


def _retrieve_batch_results(client: Any, batch_id: str) -> Dict[str, Any]:
    """Retrieve and parse all results from a completed batch.
    
    Returns a dict mapping custom_id to result object.
    """
    results = {}
    
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        results[custom_id] = result.model_dump()
    
    return results


def main_verify_only(args: argparse.Namespace, verif_model: str) -> None:
    """Verification-only mode: read predictions from CSV and verify them using three-step method."""
    
    if not args.verify_input:
        raise ValueError("--verify_input is required when using --verify_only")
    if not os.path.isfile(args.verify_input):
        raise FileNotFoundError(f"Input file not found: {args.verify_input}")
    
    print(f"Loading predictions from {args.verify_input}...", file=sys.stderr)
    
    # Read input CSV
    rows_to_verify = []
    try:
        with open(args.verify_input, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                case_prompt = row.get("case_prompt") or ""
                predicted_diagnosis = row.get("predicted_diagnosis") or ""
                true_diagnosis = row.get("true_diagnosis") or ""
                
                if not case_prompt or not predicted_diagnosis:
                    print(f"Skipping row {row_idx}: missing case_prompt or predicted_diagnosis", file=sys.stderr)
                    continue
                
                rows_to_verify.append({
                    "row_idx": row_idx,
                    "original_row": row,
                    "case_prompt": case_prompt,
                    "predicted_diagnosis": predicted_diagnosis,
                    "true_diagnosis": true_diagnosis,
                })
    except Exception as exc:
        raise RuntimeError(f"Failed to read input CSV: {exc}") from exc
    
    total_rows = len(rows_to_verify)
    if total_rows == 0:
        print("No rows to verify in input file.", file=sys.stderr)
        return
    
    print(f"Loaded {total_rows} rows for verification", file=sys.stderr)
    
    # Prepare CSV output fieldnames: preserve original columns and add verification columns
    sample_original_row = rows_to_verify[0]["original_row"]
    original_fieldnames = list(sample_original_row.keys()) if sample_original_row else []
    
    verification_fieldnames = [
        "true_diagnosis_description",
        "predicted_diagnosis_description",
        "verification_similarity",
        "verified_correct",
        "verification_response",
    ]
    
    # Combine: original fields + verification fields
    output_fieldnames = original_fieldnames + verification_fieldnames
    
    # Process rows and verify
    def verify_single_row(record: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a single row using three-step verification."""
        try:
            is_correct, true_desc, pred_desc, sim_score, rating_raw = verify_three_step(
                case_prompt=record["case_prompt"],
                predicted_diagnosis=record["predicted_diagnosis"],
                true_diagnosis=record["true_diagnosis"] or "Unknown",
                model=verif_model,
            )
            verification = (
                f"True diagnosis description:\n{true_desc}\n\n"
                f"Predicted diagnosis description:\n{pred_desc}\n\n"
                f"Similarity rating (0-10): {rating_raw}"
            )
            
            if args.sleep > 0:
                time.sleep(args.sleep)
            
            result = {
                **record["original_row"],
                "true_diagnosis_description": true_desc,
                "predicted_diagnosis_description": pred_desc,
                "verification_similarity": f"{sim_score:.2f}",
                "verified_correct": bool(is_correct),
                "verification_response": verification,
            }
            return result
        except Exception as exc:
            print(f"Error verifying row {record['row_idx']}: {exc}", file=sys.stderr)
            # Return original row with error indicators
            return {
                **record["original_row"],
                "true_diagnosis_description": "ERROR",
                "predicted_diagnosis_description": "ERROR",
                "verification_similarity": "ERROR",
                "verified_correct": False,
                "verification_response": str(exc),
            }
    
    start_time = time.perf_counter()
    processed = 0
    
    # Write results to output CSV
    file_exists = os.path.isfile(args.output)
    needs_header = not file_exists or os.path.getsize(args.output) == 0
    
    with open(args.output, "a", encoding="utf-8", newline="") as outfile, ThreadPoolExecutor(max_workers=args.workers) as executor:
        writer = csv.DictWriter(outfile, fieldnames=output_fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        
        with tqdm(total=total_rows, desc="Verifying", unit="row") as progress:
            futures = [executor.submit(verify_single_row, record) for record in rows_to_verify]
            for future in as_completed(futures):
                output_record = future.result()
                writer.writerow(output_record)
                outfile.flush()
                processed += 1
                progress.update(1)
    
    elapsed = time.perf_counter() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    print(
        f"Completed verifying {processed} rows in {format_duration(elapsed)} "
        f"({rate:.2f} rows/sec). Results saved to {args.output}"
    )
    
    # Compute and report accuracy
    try:
        correct_count = 0
        total_count = 0
        with open(args.output, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = str(row.get("verified_correct", "")).strip().lower()
                if v in {"true", "1", "yes", "y"}:
                    correct_count += 1
                total_count += 1
        
        if total_count > 0:
            accuracy = correct_count / total_count
            acc_lo, acc_hi = _wilson_ci(correct_count, total_count)
            print(f"Overall accuracy: {correct_count}/{total_count} = {accuracy:.2%} (95% CI: {acc_lo:.2%} – {acc_hi:.2%})")
    except Exception as exc:
        print(f"Warning: failed to compute accuracy: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()

    # Determine provider and models
    provider = args.provider
    # Normalize provider aliases
    if provider == "claude":
        provider = "anthropic"
    gen_model = args.gen_model or (
        "deepseek-reasoner" if provider == "deepseek" else (
            "grok-3" if provider == "xai" else "claude-sonnet-3-7"
        )
    )
    # Use gpt-5-nano for verification across providers by default
    verif_model = args.verif_model or "gpt-5-nano"

    # Initialize client settings by provider
    global CLIENT_SETTINGS, CLIENT_PROVIDER  # noqa: PLW0603
    CLIENT_PROVIDER = provider
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set")
        CLIENT_SETTINGS = {"api_key": api_key, "base_url": "https://api.deepseek.com"}
    elif provider == "xai":
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise EnvironmentError("XAI_API_KEY environment variable is not set")
        # xai-sdk client accepts api_key and optional timeout
        CLIENT_SETTINGS = {"api_key": api_key, "timeout": 3600}
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is not set")
        CLIENT_SETTINGS = {"api_key": api_key}
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Handle verification-only mode
    if args.verify_only:
        return main_verify_only(args, verif_model)

    # Show exactly how the prompt was edited (if requested)
    if (args.prompt_edit or "").strip():
        expanded = _expand_prompt_edit(args.prompt_edit)
        print("\n=== Prompt edit active ===", file=sys.stderr)
        print(f"Raw prompt_edit: {args.prompt_edit}", file=sys.stderr)
        print("Expanded insert:", file=sys.stderr)
        print(expanded, file=sys.stderr)
        print("=== End prompt edit ===\n", file=sys.stderr)

    # For skip logic, prefer multi-sample aware loader when output is CSV
    if args.skip_existing and args.output.lower().endswith(".csv"):
        existing_records_multi = load_existing_results_multi(args.output)
        existing_records = {k: v["rows"][-1] if v.get("rows") else {} for k, v in existing_records_multi.items()}
    else:
        existing_records_multi = {}
        existing_records = load_existing_results(args.output) if args.skip_existing else {}

    def build_task(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        record_id = example.get("pmcid")
        key = str(record_id) if record_id is not None else ""
        if args.skip_existing and key:
            # If multi-sample aware existing loaded, respect per-sample counts
            if key in existing_records_multi:
                have_idxs = existing_records_multi[key].get("sample_indices", set())
                missing = [i for i in range(max(0, args.samples)) if i not in have_idxs]
                if not missing:
                    return None
            else:
                # Fallback: if we have any record for this pmcid and samples>=1, skip entirely
                if key in existing_records:
                    return None

        case_prompt = example.get("case_prompt") or example.get("case_presentation")
        if not case_prompt:
            print(f"Skipping example without case prompt (id={record_id})", file=sys.stderr)
            return None

        task_base = {
            "pmcid": record_id,
            "case_prompt": case_prompt,
            "true_diagnosis": example.get("final_diagnosis"),
            # Assumption: dataset provides a field named 'diagnostic_reasoning'
            # If absent, this will be None and written as empty in CSV
            "diagnostic_reasoning": example.get("diagnostic_reasoning"),
        }

        # Expand into per-sample tasks (default to one)
        n = max(1, args.samples)
        if args.skip_existing and key in existing_records_multi:
            have_idxs = existing_records_multi[key].get("sample_indices", set())
            indices = [i for i in range(n) if i not in have_idxs]
        else:
            indices = list(range(n))

        if not indices:
            return None

        # Return a sentinel dict containing a list of expanded tasks; caller will handle expansion
        return {"_expanded": [{**task_base, "sample_index": i} for i in indices]}

    # Load and optionally sample cases
    all_examples = list(iter_dataset(args.dataset, args.split, args.limit))
    include_ids = [s.strip() for s in (args.include_pmcids.split(",") if args.include_pmcids else []) if s.strip()]
    if args.num_cases and args.num_cases > 0:
        selected_examples = select_examples(all_examples, args.num_cases, include_ids, seed=args.seed)
    else:
        selected_examples = all_examples

    tasks = []
    for example in selected_examples:
        prepared = build_task(example)
        if prepared is None:
            continue
        if "_expanded" in prepared:
            tasks.extend(prepared["_expanded"])  # list of per-sample tasks
        else:
            # Single task (samples == 1 and not skipping existing)
            tasks.append(prepared)

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No examples to process.")
        return

    start_time = time.perf_counter()
    processed = 0

    # Prepare CSV field order
    csv_fieldnames = [
        "pmcid",
        "sample_index",
        "prompt_edit",
        "prompt_insert",
        "case_prompt",
        "diagnostic_reasoning",
        "true_diagnosis",
        "predicted_diagnosis",
        "reasoning_trace",
        "posthoc_reasoning_trace",
        "verification_response",
        "verification_similarity",
        "verified_correct",
    ]

    def process_single(task: Dict[str, Any]) -> Dict[str, Any]:
        client = get_client()
        prompt = render_generation_prompt(task["case_prompt"], args.prompt_edit)
        if provider == "deepseek":
            reasoner_result = call_deepseek_reasoner(client, prompt, model=gen_model)
        elif provider == "xai":
            reasoner_result = call_xai_reasoner(client, prompt, model=gen_model)
        else:  # anthropic
            reasoner_result = call_anthropic_reasoner(client, prompt, model=gen_model)
        final_content = reasoner_result.get("content") or ""
        final_content = final_content.replace(prompt, "").strip()
        predicted = extract_answer(final_content) or final_content.strip()
        posthoc_think = extract_think(final_content) or ""

        # Unified three-step verification using OpenAI (gpt-5-nano)
        is_correct, true_desc, pred_desc, sim_score, rating_raw = verify_three_step(
            case_prompt=task["case_prompt"],
            predicted_diagnosis=predicted,
            true_diagnosis=task["true_diagnosis"] or "Unknown",
            model=verif_model,
        )
        verification = (
            f"True diagnosis description:\n{true_desc}\n\n"
            f"Predicted diagnosis description:\n{pred_desc}\n\n"
            f"Similarity rating (0-10): {rating_raw}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

        return {
            "pmcid": str(task["pmcid"]) if task.get("pmcid") is not None else "",  # normalize ID as string for CSV
            "sample_index": int(task.get("sample_index", 0)),
            "prompt_edit": args.prompt_edit or "",
            "prompt_insert": _expand_prompt_edit(args.prompt_edit) if args.prompt_edit else "",
            "case_prompt": task["case_prompt"],
            "diagnostic_reasoning": task.get("diagnostic_reasoning"),
            "true_diagnosis": task["true_diagnosis"],
            "predicted_diagnosis": predicted,
            "reasoning_trace": reasoner_result.get("reasoning"),
            "posthoc_reasoning_trace": posthoc_think,
            "verification_response": verification,
            "verification_similarity": f"{sim_score:.2f}",
            "verified_correct": bool(is_correct),
        }

    is_csv = args.output.lower().endswith(".csv")

    # Use batch processing for Anthropic provider to reduce costs by 50%
    if provider == "anthropic":
        print(f"Using Anthropic Message Batches API for cost optimization...", file=sys.stderr)
        results = process_anthropic_batches(tasks, gen_model, verif_model, args.prompt_edit)
        processed = len(results)
        
        if is_csv:
            file_exists = os.path.isfile(args.output)
            needs_header = not file_exists or os.path.getsize(args.output) == 0
            # If appending to an existing CSV with an older header, adapt to its columns to avoid misalignment
            fieldnames_to_use = csv_fieldnames
            if file_exists and not needs_header:
                try:
                    with open(args.output, "r", encoding="utf-8", newline="") as rf:
                        reader = csv.DictReader(rf)
                        if reader.fieldnames:
                            fieldnames_to_use = reader.fieldnames
                except Exception:
                    pass
            
            with open(args.output, "a", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames_to_use, extrasaction="ignore")
                if needs_header:
                    writer.writeheader()
                for record in results:
                    writer.writerow(record)
                outfile.flush()
        else:
            with open(args.output, "a", encoding="utf-8") as outfile:
                for record in results:
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                outfile.flush()
    elif is_csv:
        # CSV path: append with header if file is new/empty
        file_exists = os.path.isfile(args.output)
        needs_header = not file_exists or os.path.getsize(args.output) == 0

        # If appending to an existing CSV with an older header, adapt to its columns to avoid misalignment
        fieldnames_to_use = csv_fieldnames
        if file_exists and not needs_header:
            try:
                with open(args.output, "r", encoding="utf-8", newline="") as rf:
                    reader = csv.DictReader(rf)
                    if reader.fieldnames:
                        fieldnames_to_use = reader.fieldnames
            except Exception:
                pass

        with open(args.output, "a", encoding="utf-8", newline="") as outfile, ThreadPoolExecutor(max_workers=args.workers) as executor:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames_to_use, extrasaction="ignore")
            if needs_header:
                writer.writeheader()

            with tqdm(total=total_tasks, desc="Processing", unit="case") as progress:
                futures = [executor.submit(process_single, task) for task in tasks]
                for future in as_completed(futures):
                    output_record = future.result()
                    writer.writerow(output_record)
                    outfile.flush()
                    processed += 1
                    progress.update(1)
    else:
        # Legacy JSONL path for backward compatibility
        with open(args.output, "a", encoding="utf-8") as outfile, ThreadPoolExecutor(max_workers=args.workers) as executor:
            with tqdm(total=total_tasks, desc="Processing", unit="case") as progress:
                futures = [executor.submit(process_single, task) for task in tasks]
                for future in as_completed(futures):
                    output_record = future.result()
                    outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    outfile.flush()
                    processed += 1
                    progress.update(1)

    elapsed = time.perf_counter() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    print(
        f"Completed processing {processed} examples in {format_duration(elapsed)} "
        f"({rate:.2f} cases/sec). Results saved to {args.output}"
    )

    # Compute and report Top-1 and Top-k accuracy for this run (k = samples per case)
    try:
        # Build the set of PMCIDs and sample indices we just generated
        run_pmcs: Dict[str, set] = {}
        for t in tasks:
            pmcid_key = str(t.get("pmcid")) if t.get("pmcid") is not None else ""
            if not pmcid_key:
                continue
            try:
                si_val = int(t.get("sample_index", 0))
            except Exception:
                si_val = 0
            run_pmcs.setdefault(pmcid_key, set()).add(si_val)

        # Collect rows for this run grouped by pmcid
        rows_by_case: Dict[str, List[Dict[str, Any]]] = {}

        # Prefer in-memory results when available (Anthropic batch path)
        if provider == "anthropic" and "results" in locals() and isinstance(results, list):
            for row in results:
                pmcid_key = str(row.get("pmcid") or "")
                if not pmcid_key or pmcid_key not in run_pmcs:
                    continue
                try:
                    si_val = int(row.get("sample_index") or 0)
                except Exception:
                    si_val = 0
                if si_val not in run_pmcs[pmcid_key]:
                    continue
                rows_by_case.setdefault(pmcid_key, []).append(row)
        else:
            # Fallback: read from CSV and filter to this run's cases and sample indices
            if os.path.isfile(args.output) and args.output.lower().endswith(".csv"):
                with open(args.output, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        pmcid_key = str(row.get("pmcid") or row.get("id") or "")
                        if not pmcid_key or pmcid_key not in run_pmcs:
                            continue
                        try:
                            si_val = int(row.get("sample_index") or 0)
                        except Exception:
                            si_val = 0
                        if si_val not in run_pmcs[pmcid_key]:
                            continue
                        rows_by_case.setdefault(pmcid_key, []).append(row)

        n_cases = len(rows_by_case)
        top1_correct = 0
        topk_correct = 0

        def _is_true(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in {"true", "1", "y", "yes"}

        for pmcid_key, rows in rows_by_case.items():
            # Determine top-1 as the smallest sample_index we produced for this case in this run
            try:
                min_si = min(run_pmcs[pmcid_key])
            except Exception:
                min_si = 0

            r_top1: Optional[Dict[str, Any]] = None
            for r in rows:
                try:
                    si_val = int(r.get("sample_index") or 0)
                except Exception:
                    si_val = 0
                if si_val == min_si:
                    r_top1 = r
                    break
            if r_top1 and _is_true(r_top1.get("verified_correct", False)):
                top1_correct += 1

            # Top-k: any of the samples for this case is correct
            if any(_is_true(r.get("verified_correct", False)) for r in rows):
                topk_correct += 1

        def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
            if n <= 0:
                return (0.0, 0.0)
            p = k / n
            denom = 1 + z*z/n
            center = (p + z*z/(2*n)) / denom
            rad = (z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)) / denom
            low = max(0.0, center - rad)
            high = min(1.0, center + rad)
            return (low, high)

        if n_cases > 0:
            top1_acc = top1_correct / n_cases
            top1_lo, top1_hi = _wilson_ci(top1_correct, n_cases)
            # Determine k for display; if uniform across cases, show the single k, else mark variable
            k_values = sorted(len(run_pmcs[c]) for c in rows_by_case.keys())
            uniform_k = k_values[0] if k_values and all(k == k_values[0] for k in k_values) else None
            topk_acc = topk_correct / n_cases
            topk_lo, topk_hi = _wilson_ci(topk_correct, n_cases)
            print(
                f"Top-1 accuracy: {top1_correct}/{n_cases} = {top1_acc:.2%} (95% CI: {top1_lo:.2%} – {top1_hi:.2%})"
            )
            if uniform_k is not None:
                print(
                    f"Top-{uniform_k} accuracy: {topk_correct}/{n_cases} = {topk_acc:.2%} (95% CI: {topk_lo:.2%} – {topk_hi:.2%})"
                )
            else:
                print(
                    f"Top-k accuracy (k varies per case): {topk_correct}/{n_cases} = {topk_acc:.2%} (95% CI: {topk_lo:.2%} – {topk_hi:.2%})"
                )
    except Exception as exc:
        print(f"Warning: failed to compute top-1/top-k accuracy: {exc}", file=sys.stderr)

    # If we produced multiple samples, generate a per-case aggregated CSV with accuracy
    def derive_aggregate_path(path: str) -> str:
        if path.lower().endswith(".csv"):
            return re.sub(r"\.csv$", ".per_case.csv", path, flags=re.IGNORECASE)
        return path + ".per_case.csv"

    if is_csv and args.samples and args.samples > 1:
        per_case: Dict[str, Dict[str, Any]] = {}
        # Read all rows from the just-written CSV
        try:
            with open(args.output, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("pmcid") or row.get("id") or "").strip()
                    if not key:
                        continue
                    bucket = per_case.setdefault(key, {
                        "pmcid": key,
                        "true_diagnosis": row.get("true_diagnosis"),
                        "total_samples": 0,
                        "n_correct": 0,
                    })
                    # Use the first non-empty true diagnosis encountered
                    if not bucket.get("true_diagnosis") and row.get("true_diagnosis"):
                        bucket["true_diagnosis"] = row.get("true_diagnosis")
                    bucket["total_samples"] += 1
                    v = str(row.get("verified_correct")).strip().lower()
                    is_ok = v in {"true", "1", "yes", "y"}
                    bucket["n_correct"] += 1 if is_ok else 0
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Failed to build per-case aggregate: {exc}", file=sys.stderr)
            return

        # Compute accuracy and write aggregate CSV
        agg_path = derive_aggregate_path(args.output)
        agg_fields = ["pmcid", "true_diagnosis", "total_samples", "n_correct", "accuracy", "accuracy_pct"]
        try:
            with open(agg_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=agg_fields)
                writer.writeheader()
                for _, rec in sorted(per_case.items(), key=lambda kv: kv[0]):
                    ts = int(rec["total_samples"]) or 1
                    nc = int(rec["n_correct"]) if rec.get("n_correct") is not None else 0
                    acc = nc / ts
                    writer.writerow({
                        "pmcid": rec.get("pmcid", ""),
                        "true_diagnosis": rec.get("true_diagnosis", ""),
                        "total_samples": ts,
                        "n_correct": nc,
                        "accuracy": f"{acc:.3f}",
                        "accuracy_pct": f"{acc*100:.1f}",
                    })
            print(f"Per-case aggregate saved to {agg_path}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Failed to write per-case aggregate CSV: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
