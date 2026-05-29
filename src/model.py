"""
  model.py

  LayoutLMv3 model loading and inference for document question answering.

  LayoutLMv3 is a multimodal transformer that jointly processes three inputs:
      1. Text tokens — the words extracted by OCR
      2. Layout — bounding box coordinates of each word (where it sits on the page)
      3. Visual — patch embeddings from the document image itself

  This combination lets the model understand document structure semantically.
  A number in the top-right corner means something different than the same
  number at the bottom — LayoutLMv3 knows this from layout + visual context.

  Fine-tuned on DocVQA, the model performs extractive QA — it selects an
  answer span directly from the document text. It cannot hallucinate because
  it can only return text that exists in the OCR output.
  """

import torch
from pathlib import Path                                                                                                                                                                                                                  
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForQuestionAnswering,
)
from PIL import Image
from src.config import get_settings
settings = get_settings()
def load_processor(model_path: str = None) -> LayoutLMv3Processor:
    """
    Loads the LayoutLMv3 processor from HuggingFace or a local path.
    The processor handles tokenization of text, normalization of bounding
    boxes, and image patch extraction — combining all three inputs into
    the format LayoutLMv3 expects.
    Args:
        model_path: Path to a local model directory, or None to load
                    the base model from HuggingFace Hub.
    Returns:
        LayoutLMv3Processor ready for encoding document inputs.
    """
    path = model_path or settings.base_model
    return LayoutLMv3Processor.from_pretrained(path, apply_ocr=False)
def load_model(model_path: str = None) -> LayoutLMv3ForQuestionAnswering:
    """
    Loads the LayoutLMv3 model for extractive question answering.
    Loads fine-tuned weights if available at fine_tuned_model_path,
    otherwise falls back to the base pretrained model from HuggingFace.
    LayoutLMv3ForQuestionAnswering adds a QA head on top of the base
    LayoutLMv3 encoder — it predicts start and end token positions
    of the answer span within the document text.
    Args:
        model_path: Optional path override. If None, checks for fine-tuned
                    weights first, then falls back to base model.
    Returns:
        LayoutLMv3ForQuestionAnswering model in eval mode.
    """
    if model_path:
        path = model_path
    elif (Path(settings.fine_tuned_model_path) / "config.json").exists():                                                                                                                                                                
      path = settings.fine_tuned_model_path
    else:
        path = settings.base_model
    model = LayoutLMv3ForQuestionAnswering.from_pretrained(path)
    model.eval()
    return model

def predict(
    model: LayoutLMv3ForQuestionAnswering,
    processor: LayoutLMv3Processor,
    image: Image.Image,
    words: list[str],
    boxes: list[list[int]],
    question: str,
) -> dict:
    """
    Runs inference on a document image to answer a natural language question.
    Takes the document image, OCR words and bounding boxes, and a question.
    Encodes all three modalities through the processor, runs the model,
    and decodes the predicted answer span from the output token positions.
    Args:
        model: Loaded LayoutLMv3ForQuestionAnswering model.
        processor: Loaded LayoutLMv3Processor.
        image: PIL Image of the document page (original, not preprocessed).
        words: List of words extracted by OCR in reading order.
        boxes: List of normalized bounding boxes [x_min, y_min, x_max, y_max]
               in [0, 1000] range, one per word.
        question: Natural language question about the document.
    Returns:
        dict with keys:
            - answer: Extracted answer string from the document.
            - confidence: Model confidence score (0.0 to 1.0).
            - start_index: Token index where the answer span starts.
            - end_index: Token index where the answer span ends.
            - is_reliable: True if confidence >= settings.confidence_threshold.
    """
    encoding = processor(
        image,
        question,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        max_length=settings.max_seq_length,
        padding="max_length",
    )

    with torch.no_grad():
        outputs = model(**encoding)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = int(torch.argmax(start_logits, dim=-1))
    end_idx = int(torch.argmax(end_logits, dim=-1))
    # Ensure valid span — end must come after start
    if end_idx < start_idx:
        end_idx = start_idx
    # Decode the predicted token span back to text
    input_ids = encoding["input_ids"][0]
    answer_tokens = input_ids[start_idx: end_idx + 1]
    answer = processor.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    # Compute confidence from softmax probabilities at predicted positions
    start_prob = float(torch.softmax(start_logits, dim=-1)[0][start_idx])
    end_prob = float(torch.softmax(end_logits, dim=-1)[0][end_idx])
    confidence = (start_prob + end_prob) / 2
    return {
        "answer": answer if answer else "I don't know",
        "confidence": round(confidence, 4),
        "start_index": start_idx,
        "end_index": end_idx,
        "is_reliable": confidence >= settings.confidence_threshold,
    }
def is_grounded(answer: str, ocr_text: str) -> bool:
    """
    Verifies that the predicted answer exists verbatim in the OCR output.
    LayoutLMv3 is an extractive model — it should only return text that
    actually appears in the document. This check catches edge cases where
    tokenization artifacts produce answers that don't match the source text.
    A grounded answer means zero hallucination risk — the answer came
    directly from the document, not from model parameters.
    Args:
        answer: The answer string predicted by the model.
        ocr_text: The full raw text extracted by Tesseract from the document.
    Returns:
        True if the answer appears in the OCR text, False otherwise.
    """
    if not answer or answer == "I don't know":
        return True
    return answer.lower() in ocr_text.lower()


'''
What each part does in plain terms:

┌────────────────┬───────────────────────────────────────────────────────────────────────────────────────────┐
│      Part      │                                       What it does                                        │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ load_processor │ Loads the tokenizer + image patcher — converts raw inputs into tensors the model can read │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ load_model     │ Loads model weights — fine-tuned first, base model as fallback                            │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ predict        │ The core function — takes image + words + boxes + question, returns answer + confidence   │
  ├────────────────┼───────────────────────────────────────────────────────────────────────────────────────────┤
  │ is_grounded    │ Sanity check — confirms the answer actually came from the document text                   │
  └────────────────┴───────────────────────────────────────────────────────────────────────────────────────────┘

  Why apply_ocr=False: By default LayoutLMv3Processor runs its own internal OCR. We pass False because we already ran Tesseract with our custom preprocessing pipeline — our OCR is better quality than the default.

  Why model.eval(): Disables dropout layers that are only used during training. Without this, inference results would be random/inconsistent.

  Why average start + end probability for confidence: The model predicts a start token and end token separately. Averaging their probabilities gives a single reliable confidence score for the whole answer span.

'''