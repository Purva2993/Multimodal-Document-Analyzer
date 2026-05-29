# Multimodal Document Analyzer

Document question answering using LayoutLMv3 — understands text, layout, and visual structure simultaneously. Upload any document (invoice, form, contract, memo) and ask questions in plain English.

## What it does

- Extracts text and bounding box coordinates from documents using Tesseract OCR
- Preprocesses images through an OpenCV pipeline (deskew, denoise, binarize, contrast enhancement)
- Answers natural language questions using LayoutLMv3 fine-tuned on DocVQA
- Stores documents as vector embeddings in ChromaDB for multi-document semantic search
- Exposes all functionality via a FastAPI REST backend
- Provides a three-tab Gradio UI for asking questions, managing a document library, and running benchmarks

## Why LayoutLMv3

Standard NLP models read text only. LayoutLMv3 jointly encodes three modalities:

- **Text** — the words extracted by OCR
- **Layout** — bounding box coordinates of each word (position on the page)
- **Visual** — image patch embeddings from the document itself

This means a number in the top-right corner is understood differently than the same number at the bottom — because the model knows where words sit, not just what they say.

## Architecture

```
PDF / Image
    ↓
OpenCV preprocessing (deskew → denoise → binarize → contrast)
    ↓
Tesseract OCR (words + bounding boxes normalized to [0, 1000])
    ↓
ChromaDB (store page embeddings for multi-document search)
    ↓
ChromaDB semantic search (find most relevant page for the question)
    ↓
LayoutLMv3 inference (extractive QA — answer span from document text)
    ↓
Groundedness check (verify answer exists verbatim in OCR output)
    ↓
FastAPI response → Gradio UI
```

## Project structure

```
multimodal-document-analyzer/
├── src/
│   ├── config.py          # Pydantic settings loaded from .env
│   ├── image_processor.py # OpenCV preprocessing pipeline
│   ├── ocr.py             # Tesseract OCR + CER metric
│   ├── model.py           # LayoutLMv3 load + inference
│   ├── dataset.py         # DocVQA dataset + PyTorch Dataset class
│   ├── train.py           # Fine-tuning loop with early stopping
│   ├── pipeline.py        # End-to-end orchestrator
│   ├── evaluate.py        # EM, F1, CER, groundedness, latency
│   └── rag.py             # ChromaDB storage and retrieval
├── api/
│   └── main.py            # FastAPI backend (7 endpoints)
├── app/
│   └── gradio_app.py      # Gradio 3-tab UI
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   └── fine_tuned/        # Saved model weights after training
├── data/
│   └── chroma_db/         # Persistent vector store
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites

- Python 3.11
- Tesseract OCR installed system-wide
- HuggingFace account and access token

Install Tesseract on Mac:
```bash
brew install tesseract
```

### Installation

```bash
git clone https://github.com/yourusername/multimodal-document-analyzer.git
cd multimodal-document-analyzer

python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and set your HuggingFace token:
```
HUGGINGFACE_TOKEN=hf_your_token_here
CONFIDENCE_THRESHOLD=0.3
BATCH_SIZE=1
MAX_SEQ_LENGTH=256
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept the dataset terms at [huggingface.co/datasets/nielsr/docvqa_1200_examples](https://huggingface.co/datasets/nielsr/docvqa_1200_examples).

## Training

### On your local machine (Mac)

```bash
python -c "from src.train import train; train()"
```

Training on Apple Silicon (MPS): ~36 minutes per epoch at batch_size=1, max_seq_length=256.

### On Google Colab (recommended — GPU)

1. Zip the project folder and upload to Google Drive
2. Open a new Colab notebook, set runtime to T4 GPU
3. Run:

```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os
with zipfile.ZipFile('/content/drive/MyDrive/multimodal-document-analyzer.zip', 'r') as z:
    z.extractall('/content/multimodal-document-analyzer')

os.chdir('/content/multimodal-document-analyzer/multimodal-document-analyzer')

!pip install -q torch transformers datasets accelerate evaluate timm \
    pytesseract opencv-python-headless PyMuPDF Pillow \
    sentence-transformers chromadb fastapi uvicorn python-multipart \
    gradio pandas scikit-learn python-dotenv pydantic pydantic-settings \
    requests huggingface-hub
```

```python
import os
os.environ['HUGGINGFACE_TOKEN'] = 'your_token_here'
os.environ['BASE_MODEL'] = 'microsoft/layoutlmv3-base'
os.environ['FINE_TUNED_MODEL_PATH'] = './models/fine_tuned'
os.environ['CONFIDENCE_THRESHOLD'] = '0.3'
os.environ['DATASET_NAME'] = 'lmms-lab/DocVQA'
os.environ['NUM_EPOCHS'] = '3'
os.environ['BATCH_SIZE'] = '8'
os.environ['MAX_SEQ_LENGTH'] = '512'
os.environ['LEARNING_RATE'] = '5e-5'

from src.train import train
results = train()
print(results)
```

Training on Colab T4: ~15 minutes for 3 epochs at batch_size=8.

After training, download the model:
```python
import shutil
from google.colab import files
shutil.make_archive('/content/fine_tuned_model', 'zip', './models/fine_tuned')
files.download('/content/fine_tuned_model.zip')
```

Unzip and replace your local `models/fine_tuned/` with the downloaded weights.

## Running the app

**Terminal 1 — API:**
```bash
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Gradio UI:**
```bash
source venv/bin/activate
python app/gradio_app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

### Using Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
```

API at [http://localhost:8000](http://localhost:8000), UI at [http://localhost:7860](http://localhost:7860).

## API endpoints

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | API status and model config |
| POST | `/ask` | Upload document + question → answer |
| POST | `/process-doc` | Upload document → OCR + store in ChromaDB |
| POST | `/search` | Semantic search across stored documents |
| GET | `/documents` | List all stored documents |
| POST | `/delete` | Remove a document from ChromaDB |
| POST | `/evaluate` | Run DocVQA benchmark (EM, F1, CER, latency) |

Interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

## Evaluation metrics

| Metric | What it measures |
|---|---|
| Exact Match (EM) | % of answers word-for-word correct |
| F1 Score | Word overlap between predicted and correct answer |
| Character Error Rate (CER) | OCR quality at character level |
| Groundedness | % of answers found verbatim in document text |
| Avg latency | Seconds per document end-to-end |

Run evaluation from the Gradio UI (Evaluate tab) or directly:
```python
from src.evaluate import run_evaluation
results = run_evaluation(num_samples=50)
```

## Key design decisions

**Why LayoutLMv3 over a text-only model**
Text-only models lose layout information when PDFs are converted to plain text. A number in a "Total" row means something different than the same number in a "Subtotal" row — LayoutLMv3 understands this because it encodes position alongside text.

**Why extractive QA over generative**
LayoutLMv3 selects an answer span directly from the document — it cannot return text that doesn't exist in the OCR output. This eliminates hallucination by design, which is critical for business documents where accuracy matters.

**Why Gradio over Streamlit**
Gradio is HuggingFace-native, built specifically for ML model demos. It has built-in components for document upload and confidence score display, and is the standard tool for demos on HuggingFace Spaces.

**Why ChromaDB for multi-document search**
Without ChromaDB, every question requires scanning every document. ChromaDB stores dense vector embeddings for each page — a semantic search finds the most relevant page in milliseconds regardless of how many documents are stored.

**Why confidence threshold at 0.3**
The model returns a confidence score for every answer. Below 0.3 the model returns "I don't know" rather than a potentially wrong answer. 0.3 was chosen based on observed score distributions — correct answers tend to score above this threshold.

## Known limitations

- **One page at a time** — LayoutLMv3 processes single pages. Questions that require reasoning across multiple pages are not supported. ChromaDB mitigates this by finding the most relevant page before running inference.
- **Training data size** — Fine-tuned on ~5,000 DocVQA samples. Production accuracy requires the full 39,000+ sample dataset. The pipeline fully supports larger datasets.
- **OCR dependency** — Model accuracy is bounded by OCR quality. Rotated, low-resolution, or handwritten documents will reduce accuracy.
- **English only** — Tesseract is configured for English. Multi-language support requires additional Tesseract language packs and retraining.

## Interview talking points

**On the architecture:**
"LayoutLMv3 jointly encodes text, layout, and visual information. The layout encoding — bounding box coordinates normalized to a 0-1000 grid — is what makes it understand document structure rather than just reading words left to right."

**On extractive vs generative:**
"I chose extractive QA deliberately. The model selects a span from the OCR output — it cannot generate text that isn't in the document. For business documents this is the right tradeoff: no hallucination, at the cost of not being able to synthesize or summarize."

**On training limitations:**
"I fine-tuned on approximately 5,000 DocVQA samples due to hardware constraints. The model demonstrates correct span extraction behavior and generalizes to unseen documents. Full accuracy requires the complete 39,000-sample dataset, which the training pipeline supports — it's a data and compute constraint, not an architectural one."

**On ChromaDB:**
"Without RAG, every question requires running inference on every page of every document — O(n) inference cost. ChromaDB reduces this to a single vector search plus one inference pass, regardless of how many documents are stored."

**Challenges**
1. Model collapse during fine-tuning
  
  Training loss dropped to 0.0000 by epoch 2. The model predicted position 0 (the [CLS] special token) for every answer with 100% fake confidence, returning "I don't know" for all questions. Root cause: get_answer_token_positions used
  processor.tokenizer.encode() which silently fails on the LayoutLMv3 fast tokenizer, returning (0, 0) for almost every sample. The model learned position 0 was always "correct." Fix: replaced with
  tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answer)) and reduced from 5 to 3 epochs to prevent memorization.
  
  ---
  2. MPS out of memory on Apple Silicon
  
  Training crashed with MPS allocated: 8.73 GB, max allowed: 9.07 GB at batch_size=4, max_seq_length=512. Root cause: LayoutLMv3 attention memory scales as O(sequence_length²) — 512 tokens requires 512×512=262,144 attention values per layer across 12
  layers. Fix: reduced batch_size to 1 and max_seq_length to 256 on Mac, moved full training to Google Colab T4 GPU where batch_size=8 and max_seq_length=512 fit comfortably.

  ---
  3. Bounding box dtype mismatch on MPS
  
  RuntimeError: Expected tensor scalar types Long, Int; but got MPSFloatType during the first training batch. Root cause: LayoutLMv3's spatial position embedding layer requires integer indices. The processor returned bbox as float tensors and MPS device
  transfer did not automatically cast them. Fix: added batch["bbox"] = batch["bbox"].long() explicitly after moving the batch to device.

  ---
  4. LayoutLMv3 tokenizer incompatibility
  
  TypeError: PreTokenizedEncodeInput must be Union[PreTokenizedInputSequence] when encoding answer spans. Root cause: the LayoutLMv3 fast tokenizer is designed for pre-tokenized inputs and rejects plain string encoding. Fix: used tokenizer.tokenize() +
  tokenizer.convert_tokens_to_ids() to encode plain strings without triggering the pre-tokenization requirement.

  ---
  5. PDF pages rendered upside down
  
  Multi-page PDFs with scanned pages caused ChromaDB to select upside-down pages as most relevant, returning garbled OCR text. Root cause: PyMuPDF renders all pages regardless of orientation and ChromaDB had no way to distinguish well-OCR'd pages from
  garbled ones. Fix: added OCR confidence filtering before ChromaDB search — pages with confidence below 30% are excluded.

  ---
  6. Training dataset size misunderstanding
  
  nielsr/docvqa_1200_examples contains only 1,200 samples not 50,000+. Training on this caused date and name biases. Fix: switched to lmms-lab/DocVQA validation split (5,349 samples) on Colab T4. Best validation loss improved from 3.89 to 2.82.

  ---
  7. Grayscale images crashing the processor
  
  ValueError: Unsupported number of image dimensions: 2 — some DocVQA images are grayscale while LayoutLMv3 requires RGB. Fix: added image = image.convert("RGB") after every PIL image load in dataset.py, evaluate.py, and pipeline.py.