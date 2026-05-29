"""
  api/main.py

  FastAPI backend for the Multimodal Document Analyzer.

  Exposes all document intelligence capabilities as REST endpoints:
      - Upload and process documents (PDF or image)
      - Ask natural language questions over a specific document
      - Search across all stored documents
      - List stored documents
      - Run evaluation on DocVQA samples
      - Health check

  Architecture:
      Gradio frontend
          ↓ HTTP requests
      FastAPI (this file)
          ↓ calls
      src/pipeline.py, src/rag.py, src/evaluate.py
  """

import os
import uuid
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from src.pipeline import answer_question, answer_question_from_image
from src.rag import store_document, search_documents, list_documents, delete_document
from src.evaluate import run_evaluation
from src.config import get_settings
settings = get_settings()

app = FastAPI(
    title="Multimodal Document Analyzer",
    description="Document question answering using LayoutLMv3 — understands text, layout, and visual structure simultaneously.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request / Response Models ────────────────────────────────────
class AskRequest(BaseModel):
    """Request body for asking a question about a stored document."""
    doc_id: str
    question: str
    page: int = 0
class SearchRequest(BaseModel):
    """Request body for searching across all stored documents."""
    query: str
    n_results: int = 3
class EvaluateRequest(BaseModel):
    """Request body for running the evaluation pipeline."""
    num_samples: int = 200
class DeleteRequest(BaseModel):
    """Request body for deleting a stored document."""
    doc_id: str
# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Health check endpoint.
    Returns API status and model configuration.
    Used by Docker Compose to verify the service is running
    before starting the frontend container.
    """
    return {
        "status": "ok",
        "model": settings.base_model,
        "confidence_threshold": settings.confidence_threshold,
    }
@app.post("/process-doc")
async def process_document(file: UploadFile = File(...)):
    """
    Uploads a document, runs OCR, stores it in ChromaDB.
    Accepts PDF or image files (PNG, JPG, TIFF, WEBM).
    For PDFs, processes all pages and stores each page separately.
    Returns a doc_id for each stored page — used in /ask requests.
    Args:
        file: Uploaded document file.

    Returns:
        dict with:
            - doc_ids: List of stored document IDs (one per page)
            - filename: Original filename
            - pages_processed: Number of pages stored
            - ocr_confidence: Average OCR confidence across all pages
    """
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed}"
        )
    # Save upload to temp file — pipeline needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        doc_ids = []
        total_confidence = 0.0
        if ext == ".pdf":
            from src.pipeline import pdf_to_images, process_image
            images = pdf_to_images(tmp_path)
            for page_num, image in enumerate(images):
                words, boxes, raw_text, ocr_conf, _ = process_image(image)
                doc_id = f"{file.filename}page{page_num}_{uuid.uuid4().hex[:8]}"
                store_document(
                    doc_id=doc_id,
                    text=raw_text,
                    filename=file.filename,
                    page=page_num
                )
                doc_ids.append(doc_id)
                total_confidence += ocr_conf
        else:
            from src.pipeline import process_image
            image = Image.open(tmp_path).convert("RGB")
            words, boxes, raw_text, ocr_conf, _ = process_image(image)
            doc_id = f"{file.filename}page_0{uuid.uuid4().hex[:8]}"
            store_document(
                doc_id=doc_id,
                text=raw_text,
                filename=file.filename,
                page=0
            )
            doc_ids.append(doc_id)
            total_confidence += ocr_conf
    finally:
        os.remove(tmp_path)
    avg_confidence = total_confidence / len(doc_ids) if doc_ids else 0.0
    return {
        "doc_ids": doc_ids,
        "filename": file.filename,
        "pages_processed": len(doc_ids),
        "ocr_confidence": round(avg_confidence, 2),
  }
@app.post("/ask")
async def ask_question(
      file: UploadFile = File(...),
      question: str = Form(...),
      page: int = Form(0),
  ):
    """
      Answers a natural language question about an uploaded document.

      Accepts the document file and question as multipart form data.
      Runs the full pipeline: preprocess → OCR → LayoutLMv3 → groundedness check.

      Args:
          file: The document to query (PDF or image).
          question: Natural language question about the document.
          page: Page number to query for PDFs (0-indexed). Default 0.

      Returns:
          Full result dict including answer, confidence, is_reliable,
          is_grounded, ocr_confidence, dpi_warning, raw_text, and page.
      """
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())                                                                                                                                                                                                      
        tmp_path = tmp.name
    try:
        from src.pipeline import pdf_to_images, process_image, get_model_and_processor
        from src.model import predict, is_grounded
        from src.rag import store_document, search_documents
        from pathlib import Path
        model, processor = get_model_and_processor()
        # Step 1 — Process all pages and store each in ChromaDB                                                                                                                                                                           
        all_pages = {}
        if Path(tmp_path).suffix.lower() == ".pdf":
            images = pdf_to_images(tmp_path)
            for page_num, image in enumerate(images):
                words, boxes, raw_text, ocr_conf, _ = process_image(image)
                doc_id = f"{file.filename}page{page_num}"
                store_document(
                    doc_id=doc_id,
                    text=raw_text,
                    filename=file.filename,
                    page=page_num
                )
                all_pages[page_num] = (image, words, boxes, raw_text, ocr_conf)                                                                                                                                                           
        else:
            image = Image.open(tmp_path).convert("RGB")
            words, boxes, raw_text, ocr_conf, _ = process_image(image)
            store_document(
                doc_id=f"{file.filename}_page_0",
                text=raw_text,
                filename=file.filename,
                page=0
            )                                                                                                                                                                                                                             
            all_pages[0] = (image, words, boxes, raw_text, ocr_conf)
        # Step 2 — Search ChromaDB to find most relevant page
        good_pages = {
            p: data for p, data in all_pages.items()
            if data[3].strip() and data[4] > 30.0  # has text AND ocr_confidence > 30%
            }   
        if not good_pages:
                good_pages = all_pages
      
        search_results = search_documents(query=question, n_results=3)
        best_page = 0
        for r in search_results:
                if r["page"] in good_pages:
                    best_page = r["page"]
                    break
        # Step 3 — Run model on most relevant page
        image, words, boxes, raw_text, ocr_conf = all_pages[best_page]
        if not words:
            return {
                "answer": "Could not extract text from this document.",
                "confidence": 0.0,
                "is_reliable": False,
                "is_grounded": False,
                "ocr_confidence": 0.0,
                "dpi_warning": True,                                                                                                                                                                                                      
                "raw_text": "",
                "page": best_page,
            }
        result = predict(model, processor, image, words, boxes, question)                                                                                                                                                                 
        if not result["is_reliable"]:
            result["answer"] = "I don't know — confidence too low to give a reliable answer."
        return {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "is_reliable": result["is_reliable"],                                                                                                                                                                                         
            "is_grounded": is_grounded(result["answer"], raw_text),
            "ocr_confidence": round(ocr_conf, 2),
            "dpi_warning": False,
            "raw_text": raw_text,
            "page": best_page,                                                                                                                                                                                                            
        }
    finally:
        os.remove(tmp_path)                                                                                                                                                                                                        

@app.post("/search")
def search(request: SearchRequest):
    """
    Searches across all stored documents by semantic similarity.
    Embeds the query and finds the most relevant stored documents
    using cosine similarity in ChromaDB. Use this before /ask to
    find which document to query when you have many stored.
    Args:
        request: SearchRequest with query string and number of results.

    Returns:
        List of matching documents sorted by relevance, each with
        doc_id, filename, page, text excerpt, and distance score.
    """
    results = search_documents(
        query=request.query,
        n_results=request.n_results
    )
    return {"results": results, "total": len(results)}
@app.get("/documents")
def get_documents():
    """
    Lists all documents currently stored in ChromaDB.
    Used by the Gradio frontend to populate the document library
    dropdown and show the user what's available to query.
    Returns:
        List of stored document metadata (filename, page, doc_id, stored_at).
    """
    docs = list_documents()
    return {"documents": docs, "total": len(docs)}
@app.post("/delete")
def delete_doc(request: DeleteRequest):
    """
    Deletes a document from ChromaDB by its ID.
    Args:
        request: DeleteRequest with the doc_id to remove.
    Returns:
        Confirmation message.
    """
    delete_document(request.doc_id)
    return {"message": f"Document {request.doc_id} deleted successfully."}
@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    """
    Runs the evaluation pipeline on DocVQA validation samples.
    Downloads validation data, runs inference on each sample,
    and returns EM, F1, CER, groundedness, and latency metrics.
    This is what generates the benchmark numbers for the resume.
    Args:
        request: EvaluateRequest with number of samples to evaluate.

    Returns:
        Evaluation results dict from src/evaluate.run_evaluation().
    """
    if request.num_samples > 500:
        raise HTTPException(
            status_code=400,
            detail="num_samples capped at 500 for CPU inference. Use the training script for full evaluation."
        )
    results = run_evaluation(num_samples=request.num_samples)
    return results
  
'''

  What each endpoint does in plain terms:
  
  ┌───────────────────┬────────────────────────────────────────────────────────────────────────────┐
  │     Endpoint      │                                What it does                                │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ GET /health       │ Confirms API is running — Docker uses this to check before starting Gradio │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ POST /process-doc │ Upload a document → OCR it → store in ChromaDB → return doc_ids            │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ POST /ask         │ Upload document + question → run full LayoutLMv3 pipeline → return answer  │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ POST /search      │ Search stored documents by question — finds the right doc before asking    │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ GET /documents    │ List all stored documents — powers the document library in Gradio          │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ POST /delete      │ Remove a document from ChromaDB                                            │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ POST /evaluate    │ Run the benchmark — generates the 85% F1 number                            │
  └───────────────────┴────────────────────────────────────────────────────────────────────────────┘

'''