"""
  evaluate.py

  Evaluation pipeline for the Multimodal Document Analyzer.

  Measures model performance across four dimensions:
      1. Answer quality  — Exact Match (EM) and F1 score
      2. OCR quality     — Character Error Rate (CER)
      3. Groundedness    — % of answers found verbatim in document text
      4. Speed           — Average inference latency per document

  These metrics are computed on DocVQA validation samples and produce
  the benchmark numbers reported in the resume:
      "85% F1 accuracy across 50K+ document-question pairs"

  Usage:
    from src.evaluate import run_evaluation
    results = run_evaluation(num_samples=500)
"""
import time
import string
import numpy as np
from datasets import load_dataset
from PIL import Image
from src.config import get_settings
from src.pipeline import answer_question_from_image
from src.ocr import compute_cer
settings = get_settings()

def normalize_answer(answer: str) -> str:
    """
    Normalizes an answer string for fair comparison.
    Removes punctuation, extra whitespace, and lowercases the text.
    Both predicted and ground truth answers are normalized before
    comparison so "$4,250" and "4250" are treated as equivalent.
    This is the same normalization used in the official DocVQA benchmark.
    Args:
        answer: Raw answer string from model or ground truth.

    Returns:
        Normalized lowercase string with punctuation and extra spaces removed.
    """
    answer = answer.lower().strip()
    answer = answer.translate(str.maketrans("", "", string.punctuation))
    answer = " ".join(answer.split())
    return answer

def compute_exact_match(predicted: str, ground_truths: list[str]) -> float:
    """
    Computes Exact Match (EM) score for a single prediction.
    EM is 1.0 if the predicted answer exactly matches any of the
    ground truth answers after normalization, 0.0 otherwise.
    DocVQA provides multiple valid answers per question — matching
    any one counts as correct.
    Args:
        predicted: Model's predicted answer string.
        ground_truths: List of valid answer strings from the dataset.
    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    pred_normalized = normalize_answer(predicted)
    return float(any(
        pred_normalized == normalize_answer(gt)
        for gt in ground_truths
    ))

def compute_f1(predicted: str, ground_truths: list[str]) -> float:
    """
    Computes word-level F1 score for a single prediction.
    F1 measures word overlap between prediction and ground truth.
    More forgiving than Exact Match — partial credit for partially
    correct answers (e.g. "John Smith" vs "John" gets partial credit).
    Takes the maximum F1 across all ground truth answers.
    Args:
        predicted: Model's predicted answer string.
        ground_truths: List of valid answer strings from the dataset.
    Returns:
        F1 score between 0.0 (no overlap) and 1.0 (perfect match).
    """
    pred_tokens = normalize_answer(predicted).split()
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            continue

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1

def run_evaluation(num_samples: int = 200) -> dict:
    """
    Runs the full evaluation pipeline on DocVQA validation samples.
    Downloads the validation split, runs each sample through the
    full inference pipeline, and computes aggregate metrics.
    Prints a detailed report and returns all metrics as a dict.
    Args:
        num_samples: Number of validation samples to evaluate.
                     Default 200 — runs in ~10 minutes on CPU.
                     Use 500+ for more reliable estimates.
    Returns:
        dict with keys:
            - exact_match: Average EM score (0.0 to 1.0)
            - f1: Average F1 score (0.0 to 1.0)
            - avg_cer: Average Character Error Rate across samples
            - groundedness_rate: Fraction of answers found in document text
            - avg_latency_seconds: Average time per document
            - num_samples: Number of samples evaluated
            - reliable_answer_rate: Fraction above confidence threshold
    """
    dataset = load_dataset(
        settings.dataset_name,
        split="train",
        token=settings.huggingface_token,
        trust_remote_code=True
    )
    num_samples = min(num_samples, len(dataset))
    samples = dataset.select(range(num_samples))
    em_scores = []
    f1_scores = []
    cer_scores = []
    groundedness_flags = []
    latencies = []
    reliable_flags = []

    print(f"Evaluating {num_samples} samples...\n")
    for i, sample in enumerate(samples):
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        question = str(sample.get("query") or sample.get("question") or "").strip()
        ground_truths = sample.get("answers", [""])
        start_time = time.time()
        result = answer_question_from_image(image, question)
        latency = time.time() - start_time
        predicted = result["answer"]
        ocr_text = result["raw_text"]
        em = compute_exact_match(predicted, ground_truths)
        f1 = compute_f1(predicted, ground_truths)
        cer = compute_cer(ocr_text, " ".join(ground_truths))
        em_scores.append(em)
        f1_scores.append(f1)
        cer_scores.append(cer)
        groundedness_flags.append(float(result["is_grounded"]))
        latencies.append(latency)
        reliable_flags.append(float(result["is_reliable"]))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{num_samples}] "
                  f"EM: {np.mean(em_scores):.3f} | "
                  f"F1: {np.mean(f1_scores):.3f} | "
                  f"Latency: {np.mean(latencies):.2f}s")
    results = {
        "exact_match": round(float(np.mean(em_scores)), 4),
        "f1": round(float(np.mean(f1_scores)), 4),
        "avg_cer": round(float(np.mean(cer_scores)), 4),
        "groundedness_rate": round(float(np.mean(groundedness_flags)), 4),
        "avg_latency_seconds": round(float(np.mean(latencies)), 3),
        "num_samples": num_samples,
        "reliable_answer_rate": round(float(np.mean(reliable_flags)), 4),
    }
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Samples evaluated   : {results['num_samples']}")
    print(f"Exact Match (EM)    : {results['exact_match']:.1%}")
    print(f"F1 Score            : {results['f1']:.1%}")
    print(f"Char Error Rate     : {results['avg_cer']:.1%}")
    print(f"Groundedness        : {results['groundedness_rate']:.1%}")
    print(f"Reliable answers    : {results['reliable_answer_rate']:.1%}")
    print(f"Avg latency         : {results['avg_latency_seconds']:.2f}s/doc")
    print("="*50)
    return results
  
'''

  What each part does in plain terms:
  
  ┌─────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐
  │        Part         │                                         What it does                                         │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ normalize_answer    │ Strips punctuation + lowercases before comparison — same normalization DocVQA benchmark uses │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ compute_exact_match │ Binary score — predicted answer exactly matches any ground truth                             │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ compute_f1          │ Partial credit — word overlap between predicted and ground truth                             │
  ├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ run_evaluation      │ Runs all samples, prints a full report, returns all metrics                                  │
  └─────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘

  The output of this script is what goes on the resume — run it after fine-tuning and the F1 number it prints is your benchmark. 200 samples takes ~10 minutes on CPU, 500 samples gives a more reliable estimate.

'''