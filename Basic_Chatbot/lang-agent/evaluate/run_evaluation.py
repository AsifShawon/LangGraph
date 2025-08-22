"""Evaluation runner for the PhysicsBotAgent.

- Does not modify anything in `src/`.
- Uses the streaming API `PhysicsBotAgent.stream` to measure timings and extract THINK/ANSWER outputs.

Usage (from repo root):

  # (optional) create venv and install requirements already in repo
  python -m venv .venv
  source .venv/bin/activate    # on bash.exe
  pip install -r requirements.txt

  # run the evaluator
  python evaluate/run_evaluation.py \ 
      --questions evaluate/sample_questions.jsonl \ 
      --out evaluate/results.jsonl \ 
      --tolerance 1e-3

The script writes per-item JSONL results and prints a small summary.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from statistics import mean, median
from typing import Optional
from difflib import SequenceMatcher

# Ensure src is importable without changing src/ files
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Ensure working directory is the repository root so relative paths used in src/ resolve correctly
os.chdir(ROOT)

# Provide a lightweight shim to avoid loading FAISS/vectorstores during quick local evals.
# The shim is active by default unless EVAL_LOAD_VECTORSTORE=1 is set in the environment.
if os.environ.get("EVAL_LOAD_VECTORSTORE", "0") != "1":
    import types

    _dummy = types.ModuleType("retrieve_physics_query")

    def query_physics(q):
        # Return an empty string (no relevant context) or a small placeholder list.
        return "No relevant physics knowledge found."

    _dummy.query_physics = query_physics
    sys.modules["retrieve_physics_query"] = _dummy

# Import the agent factory from src/agent.py
try:
    from agent import create_agent
except Exception as e:
    raise ImportError(f"Failed to import create_agent from src.agent: {e}")


def extract_tagged_block(text: str, tag: str) -> str:
    """Return content inside <TAG>...</TAG> if present, otherwise return original text."""
    pattern = re.compile(rf"<{tag}>([\s\S]*?)</{tag}>", re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()


def try_parse_number(s: str) -> Optional[float]:
    """Try to extract the first numeric value from string.
    Accepts integers, decimals, scientific notation, and formatted numbers.
    Returns None if no parseable number found.
    """
    if not s:
        return None
    
    # Clean the string: remove common non-numeric characters but preserve scientific notation
    cleaned = re.sub(r'[^\d\.\-\+eE\s]', ' ', s)
    
    # Enhanced patterns for different number formats
    patterns = [
        r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?',  # Scientific notation
        r'[-+]?[0-9]+\.?[0-9]*',  # Regular decimals
        r'[-+]?[0-9]+',  # Integers
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, cleaned)
        for match in matches:
            try:
                return float(match)
            except ValueError:
                continue
    
    return None


def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts using multiple methods."""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()
    
    # Exact match
    if t1 == t2:
        return 1.0
    
    # Sequence matching (word order matters)
    seq_match = SequenceMatcher(None, t1, t2).ratio()
    
    # Word-based similarity (order doesn't matter)
    words1 = set(re.findall(r'\w+', t1))
    words2 = set(re.findall(r'\w+', t2))
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    word_intersection = len(words1.intersection(words2))
    word_union = len(words1.union(words2))
    word_similarity = word_intersection / word_union if word_union > 0 else 0.0
    
    # Combine both methods (weighted average)
    return 0.6 * seq_match + 0.4 * word_similarity


async def evaluate_one(agent, question_text: str, thread_id: str = "eval-thread"):
    """Run the agent.stream and capture THINK/ANSWER and timings.
    Returns a dict with timings, raw texts and any error.
    """
    start = time.perf_counter()
    thinking_time = None
    thinking_text = None
    answer_time = None
    answer_text = None
    error = None

    try:
        async for event in agent.stream(question_text, thread_id):
            etype = event.get("type")
            if etype == "thinking":
                if thinking_time is None:
                    thinking_time = time.perf_counter()
                    thinking_text = event.get("content", "")
            elif etype == "answer":
                # final answer
                if answer_time is None:
                    answer_time = time.perf_counter()
                    answer_text = event.get("content", "")
                    # We break after getting final answer
                    break
            elif etype == "error":
                error = event.get("content")
                break
    except Exception as e:
        error = str(e)

    end = time.perf_counter()
    result = {
        "question": question_text,
        "thinking_raw": thinking_text,
        "answer_raw": answer_text,
        "error": error,
        "timings": {
            "t_start_to_thinking": None if thinking_time is None else (thinking_time - start),
            "t_thinking_to_answer": None if (thinking_time is None or answer_time is None) else (answer_time - thinking_time),
            "t_total": end - start,
        },
    }
    return result


def score_result(result, item_data: dict, tolerance: float):
    """Score the result based on question type and using semantic similarity for conceptual questions."""
    out = {"type": "unscored", "pass": False}
    answer_raw = result.get("answer_raw") or ""
    answer_inside = extract_tagged_block(answer_raw, "ANSWER")
    out["answer_text"] = answer_inside
    
    ref_text = item_data.get("ref")
    question_type = item_data.get("type", "unknown")
    
    if question_type == "numerical":
        # Numerical scoring with improved parsing
        numeric_ref = try_parse_number(ref_text) if ref_text else None
        numeric_ans = try_parse_number(answer_inside)
        
        if numeric_ref is not None and numeric_ans is not None:
            abs_err = abs(numeric_ans - numeric_ref)
            rel_err = abs_err / abs(numeric_ref) if numeric_ref != 0 else abs_err
            # Pass if within absolute tolerance OR within 5% relative tolerance
            passes = abs_err <= tolerance or rel_err <= 0.05
            out.update({
                "type": "numeric", 
                "numeric_ref": numeric_ref, 
                "numeric_ans": numeric_ans, 
                "abs_err": abs_err,
                "rel_err": rel_err,
                "pass": passes
            })
            return out
        else:
            out.update({
                "type": "numeric_failed",
                "numeric_ref": numeric_ref,
                "numeric_ans": numeric_ans,
                "pass": False,
                "error": "Could not parse numeric answer"
            })
            return out
    
    elif question_type == "conceptual":
        # Semantic similarity scoring for conceptual questions
        if ref_text:
            similarity = semantic_similarity(answer_inside, ref_text)
            # Pass if similarity is above 0.6 (60%)
            passes = similarity >= 0.6
            out.update({
                "type": "conceptual",
                "ref": ref_text,
                "similarity": similarity,
                "pass": passes
            })
            return out
    
    # Fallback to exact text matching
    if ref_text is not None:
        match = answer_inside.strip().lower() == ref_text.strip().lower()
        out.update({"type": "text", "ref": ref_text, "match": match, "pass": match})
        return out

    out.update({"type": "text", "ref": None, "match": False, "pass": False})
    return out


async def main_async(args):
    # Load questions
    with open(args.questions, "r", encoding="utf8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    agent = create_agent()

    results = []
    for it in items:
        qid = it.get("id")
        q = it.get("q") or it.get("question")
        ref = it.get("ref")
        print(f"Running {qid}: {q[:80]}...")
        res = await evaluate_one(agent, q, thread_id=f"eval-{qid or int(time.time()*1000)}")
        score = score_result(res, it, args.tolerance)
        out = {"id": qid, "q": q, "ref": ref, "result": res, "score": score}
        results.append(out)
        
        # Show immediate result
        status = "✅ PASS" if score["pass"] else "❌ FAIL"
        if score["type"] == "numeric":
            print(f"  {status} - Expected: {score.get('numeric_ref')}, Got: {score.get('numeric_ans')}, Error: {score.get('abs_err', 'N/A'):.3f}")
        elif score["type"] == "conceptual":
            print(f"  {status} - Similarity: {score.get('similarity', 0):.2f}")
        else:
            print(f"  {status} - {score.get('type', 'unknown')}")
        
        # persist after each item
        with open(args.out, "a", encoding="utf8") as outfh:
            outfh.write(json.dumps(out) + "\n")

    # Enhanced summaries
    totals = [r["result"]["timings"]["t_total"] for r in results if r["result"]["timings"]["t_total"] is not None]
    t_mean = mean(totals) if totals else None
    t_median = median(totals) if totals else None
    t_p95 = (sorted(totals)[int(0.95 * len(totals))]) if totals else None

    # Accuracy by type
    numerical_results = [r for r in results if r["score"]["type"] in ["numeric", "numeric_failed"]]
    conceptual_results = [r for r in results if r["score"]["type"] == "conceptual"]
    
    numerical_acc = sum(1 for r in numerical_results if r["score"]["pass"]) / len(numerical_results) if numerical_results else 0
    conceptual_acc = sum(1 for r in conceptual_results if r["score"]["pass"]) / len(conceptual_results) if conceptual_results else 0
    
    acc_count = sum(1 for r in results if r["score"]["pass"]) 
    acc_rate = acc_count / len(results) if results else None

    summary = {
        "n_total": len(results),
        "n_numerical": len(numerical_results),
        "n_conceptual": len(conceptual_results),
        "overall_accuracy": acc_rate,
        "numerical_accuracy": numerical_acc,
        "conceptual_accuracy": conceptual_acc,
        "mean_latency_s": t_mean,
        "median_latency_s": t_median,
        "p95_latency_s": t_p95,
    }

    print("\n--- ENHANCED EVAL SUMMARY ---")
    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="evaluate/sample_questions.jsonl", help="JSONL file with questions: each line is {id, q, ref}")
    p.add_argument("--out", default="evaluate/results.jsonl", help="Output JSONL")
    p.add_argument("--tolerance", type=float, default=1e-3, help="Numeric tolerance for numeric answers")
    return p.parse_args()


def main():
    args = parse_args()
    # ensure out file is empty
    if os.path.exists(args.out):
        os.remove(args.out)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
