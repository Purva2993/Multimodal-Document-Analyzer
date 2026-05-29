"""

  app/gradio_app.py



  Gradio-based web interface for the Multimodal Document Analyzer.



  Provides three tabs:

      1. Ask a Document  — upload a document, ask a question, see the answer

                           with confidence score and groundedness indicator

      2. Document Library — browse all stored documents, search across them

      3. Evaluate         — run the DocVQA benchmark and see accuracy metrics



  Gradio is used instead of Streamlit because:

      - Built by HuggingFace — native fit for HuggingFace models (LayoutLMv3)

      - Native image upload and display components

      - Standard demo tool in the ML/AI ecosystem

      - No session state workarounds needed (unlike Streamlit button nesting)



  Runs on port 7860 (HuggingFace standard).

  API_URL is read from environment — defaults to localhost for local dev,

  set to http://api:8000 inside Docker via docker-compose.yml.

  """


"""

  app/gradio_app.py



  Redesigned Gradio UI for the Multimodal Document Analyzer.

  Cleaner layout, step-by-step flow, plain language labels.

  """

import os
import requests                                                                                                                                                                                                                      
import gradio as gr
API_URL = os.getenv("API_URL", "http://localhost:8000")
def check_api_health() -> str:                                                                                                                                                                                                       
    """Checks if the FastAPI backend is reachable on startup."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return "✅ Connected and ready"
        return "⚠️  API responded with an error"
    except Exception:
        return "❌ Cannot reach API — make sure uvicorn is running on port 8000"
def ask_document(file, question: str) -> tuple:
    """
    Sends uploaded document + question to the API and returns the answer.
    Args:
        file: Uploaded document file object from Gradio.
        question: Natural language question about the document.
        #page: Page number for PDFs (0 = first page).
    Returns:
        Tuple of (answer, confidence_label, groundedness, ocr_text, warning)
    """
    if file is None:
        return "⚠️  Please upload a document first.", "", "", "", ""
    if not question.strip():
        return "⚠️  Please type a question.", "", "", "", ""
    try:
        with open(file.name, "rb") as f:
            response = requests.post(
                f"{API_URL}/ask",                                                                                                                                                                                                    
                files={"file": (os.path.basename(file.name), f)},
                data={"question": question, "page": 0},
                timeout=60,
            )
        
        result = response.json()
        answer = result.get("answer", "No answer found.")
        conf = result.get("confidence", 0.0)
        reliable = result.get("is_reliable", False)
        grounded = result.get("is_grounded", False)
        if reliable:
            confidence_label = f"✅ High confidence ({conf * 100:.1f}%)"
        else:
            confidence_label = f"⚠️  Low confidence ({conf * 100:.1f}%) — treat this answer with caution"
        groundedness_label = (
            "✅ Answer found directly in the document"
            if grounded else
            "⚠️  Answer may not appear verbatim in the document"
        )
        warning = "⚠️  Low image resolution detected — OCR accuracy may be reduced." \
            if result.get("dpi_warning") else ""
        ocr_text = result.get("raw_text", "")
        page_label = f"Answer found on page {result.get('page', 0) + 1}"
        return answer, confidence_label, groundedness_label, ocr_text, warning, page_label
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", ""
    
def store_document(file) -> str:
    """
    Uploads and stores a document in the library for later search.
    Args:
        file: Uploaded document file object from Gradio.
    Returns:
        Status message confirming storage or describing the error.
    """
    if file is None:
        return "⚠️  Please upload a document first."

    try:
        with open(file.name, "rb") as f:
            response = requests.post(
                f"{API_URL}/process-doc",
                files={"file": (os.path.basename(file.name), f)},
                timeout=120,
            )
        result = response.json()
        return (                                                                                                                                                                                                                     
            f"✅ Stored '{result['filename']}' successfully\n"
            f"Pages stored: {result['pages_processed']}\n"
            f"OCR confidence: {result['ocr_confidence']}%"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"
def refresh_library() -> str:
    """
    Fetches the list of all stored documents from the API.                                                                                                                                                                           
    Returns:
        Formatted string listing all documents or a message if empty.
    """
    try:
        response = requests.get(f"{API_URL}/documents", timeout=10)
        data = response.json()
        docs = data.get("documents", [])
        if not docs:
            return "No documents stored yet.\nGo to 'Ask a Document' tab and click 'Save to Library'."
        lines = [f"📚 {data['total']} document(s) in your library:\n"]
        for doc in docs:
            lines.append(f"• {doc['filename']} — page {doc['page']} — added {doc['stored_at'][:10]}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {str(e)}"
def search_library(query: str, n_results: int) -> str:
    """
    Searches stored documents by semantic similarity to the query.
    Args:
        query: Search query or question.
        n_results: Number of top results to return.                                                                                                                                                                                  
    Returns:
        Formatted string of matching documents with similarity scores.
    """
    if not query.strip():
        return "⚠️  Please enter a search query."
    try:
        response = requests.post(
            f"{API_URL}/search",                                                                                                                                                                                                     
            json={"query": query, "n_results": int(n_results)},
            timeout=30,
        )
        data = response.json()
        results = data.get("results", [])                                                                                                                                                                                            
        if not results:
            return "No matching documents found."
        lines = [f"🔍 Top {data['total']} result(s) for: '{query}'\n"]
        for i, r in enumerate(results, 1):
            similarity = (1 - r["distance"]) * 100
            lines.append(
                f"{i}. {r['filename']} (page {r['page']}) — {similarity:.1f}% match\n"
                f"   \"{r['text'][:200]}...\"\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {str(e)}"                                                                                                                                                                                                 
def run_evaluation(num_samples: int) -> str:
    """
    Triggers the DocVQA benchmark evaluation via the API.
    Args:
        num_samples: Number of validation samples to evaluate (50–500).
    Returns:
        Formatted evaluation report with all accuracy metrics.
    """
    try:
        response = requests.post(
            f"{API_URL}/evaluate",                                                                                                                                                                                                   
            json={"num_samples": int(num_samples)},
            timeout=600,
        )
        r = response.json()
        return (
            f"📊 EVALUATION RESULTS\n"                                                                                                                                                                                               
            f"{'─' * 35}\n"
            f"Samples evaluated   : {r['num_samples']}\n"
            f"Exact Match (EM)    : {r['exact_match'] * 100:.1f}%\n"
            f"F1 Score            : {r['f1'] * 100:.1f}%\n"
            f"Character Error Rate: {r['avg_cer'] * 100:.1f}%\n"
            f"Groundedness        : {r['groundedness_rate'] * 100:.1f}%\n"
            f"Reliable Answers    : {r['reliable_answer_rate'] * 100:.1f}%\n"
            f"Avg Response Time   : {r['avg_latency_seconds']:.2f}s per document\n"
            f"{'─' * 35}"
        )
    except Exception as e:
        return f"❌ Evaluation error: {str(e)}"                                                                                                                                                                                      
def build_app() -> gr.Blocks:
    """
    Builds and returns the Gradio application with a clean three-tab layout.
    Returns:
        gr.Blocks application ready to launch.
    """
    with gr.Blocks(
        title="Multimodal Document Analyzer",
        theme=gr.themes.Soft(),
        css=".gr-button-primary { background: #2563eb !important; }"
    ) as app:
        # ── Header ───────────────────────────────────────────────
        gr.Markdown("# 📄 Multimodal Document Analyzer")
        gr.Markdown(
            "Upload any document — invoice, form, contract, report — "
            "and ask questions in plain English. Powered by LayoutLMv3."
        )
        with gr.Row():
            gr.Markdown("**API Status:**")
            gr.Markdown(check_api_health())
        gr.Markdown("---")
        with gr.Tabs():

            # ── Tab 1: Ask ────────────────────────────────────────
            with gr.Tab("📄 Ask a Document"):
                gr.Markdown("### Step 1 — Upload your document")
                file_input = gr.File(
                    label="Choose a file (PDF, PNG, JPG, TIFF)",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff"]
                )
                gr.Markdown("### Step 2 — Ask your question")
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder='e.g. "What is the invoice total?" or "Who signed the contract?"',
                    lines=2
                )
               
                gr.Markdown("### Step 3 — Get your answer")
                with gr.Row():                                                                                                                                                                                                       
                    ask_btn = gr.Button("🔍 Ask", variant="primary", scale=2)
                    save_btn = gr.Button("💾 Save to Library", scale=1)
                gr.Markdown("---")
                answer_output = gr.Textbox(                                                                                                                                                                                          
                    label="Answer",
                    lines=3,
                    interactive=False,
                    placeholder="Your answer will appear here..."
                )
                with gr.Row():                                                                                                                                                                                                       
                    confidence_output = gr.Textbox(
                        label="Confidence",
                        interactive=False
                    )
                    groundedness_output = gr.Textbox(
                        label="Groundedness",
                        interactive=False
                    )
                warning_output = gr.Textbox(
                    label="Warnings",
                    interactive=False,
                    visible=True
                )
                page_output = gr.Textbox(
                    label="Page",
                    interactive=False
                )
                save_status = gr.Textbox(
                    label="Library Status",
                    interactive=False
                )
                with gr.Accordion("📝 View full extracted text (OCR output)", open=False):
                    ocr_output = gr.Textbox(
                        label="Extracted text",
                        lines=12,
                        interactive=False
                    )
                ask_btn.click(
                    fn=ask_document,                                                                                                                                                                                                 
                    inputs=[file_input, question_input],
                    outputs=[answer_output, confidence_output,
                        groundedness_output, ocr_output, warning_output, page_output]
                )
                save_btn.click(
                    fn=store_document,
                    inputs=[file_input],                                                                                                                                                                                             
                    outputs=[save_status]
                )
            # ── Tab 2: Library ────────────────────────────────────
            with gr.Tab("📚 Document Library"):
                gr.Markdown(
                    "All documents you've saved are stored here. "
                    "Search across all of them at once."
                )
                refresh_btn = gr.Button("🔄 Refresh List", variant="primary")
                library_output = gr.Textbox(
                    label="Stored Documents",
                    lines=10,                                                                                                                                                                                                        
                    interactive=False
                )
                gr.Markdown("---")
                gr.Markdown("### Search across all documents")
                search_input = gr.Textbox(
                    label="Search query",
                    placeholder='e.g. "Project Apollo" or "termination clause"'
                )
                n_results_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Number of results to show"
                )
                search_btn = gr.Button("🔍 Search", variant="primary")
                search_output = gr.Textbox(
                    label="Search Results",
                    lines=12,
                    interactive=False
                )
                refresh_btn.click(fn=refresh_library, outputs=[library_output])
                search_btn.click(                                                                                                                                                                                                    
                    fn=search_library,
                    inputs=[search_input, n_results_slider],
                    outputs=[search_output]
                )
            # ── Tab 3: Evaluate ───────────────────────────────────                                                                                                                                                                 
            with gr.Tab("📊 Evaluate Accuracy"):
                gr.Markdown(
                    "Run the model against the DocVQA benchmark dataset "
                    "to measure accuracy. This generates the performance "
                    "numbers shown on the resume."
                )
                gr.Markdown("⏱️  **Note:** 200 samples takes ~10 minutes on CPU.")
                num_samples_slider = gr.Slider(
                    minimum=50, maximum=500, value=200, step=50,
                    label="Number of test samples"
                )                                                                                                                                                                                                                    
                eval_btn = gr.Button("▶️  Run Evaluation", variant="primary")
                eval_output = gr.Textbox(
                    label="Results",
                    lines=12,
                    interactive=False
                )

                eval_btn.click(
                    fn=run_evaluation,
                    inputs=[num_samples_slider],
                    outputs=[eval_output]
                )
    return app                                                                                                                                                                                                                       
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False                                                                                                                                                                                                                  
    )
'''
  What each part does in plain terms:



  ┌───────────────────┬─────────────────────────────────────────────────────────────────────────┐

  │       Part        │                              What it does                               │

  ├───────────────────┼─────────────────────────────────────────────────────────────────────────┤

  │ check_api_health  │ Shows API status on startup — instant feedback if backend isn't running │

  ├───────────────────┼─────────────────────────────────────────────────────────────────────────┤

  │ ask_document      │ Main function — sends document + question to API, formats the result    │

  ├───────────────────┼─────────────────────────────────────────────────────────────────────────┤

  │ process_and_store │ Stores document in ChromaDB — makes it searchable across the library    │

  ├───────────────────┼─────────────────────────────────────────────────────────────────────────┤

  │ get_document_list │ Shows all stored documents in the library tab                           │

  ├───────────────────┼─────────────────────────────────────────────────────────────────────────┤

  │ search_docs       │ Semantic search across all stored documents                             │

  ├───────────────────┼─────────────────────────────────────────────────────────────────────────┤

  │ run_evaluation_ui │ Triggers the benchmark — shows EM, F1, CER, latency in the UI           │

  ├───────────────────┼─────────────────────────────────────────────────────────────────────────┤

  │ build_app         │ Assembles the full three-tab UI with all components wired up            │

  └───────────────────┴─────────────────────────────────────────────────────────────────────────┘



  Three things worth knowing for interviews:



  1. Why server_name="0.0.0.0" — makes the app accessible from outside the container. Without this, Docker can't expose the port to your browser.

  2. Why timeout=600 on evaluate — evaluation runs 200+ inference passes. HTTP default timeouts (30s) would kill it. 600 seconds gives it room.

  3. Why Accordion for OCR text — the raw OCR output is long and clutters the UI. Collapsing it by default keeps the interface clean while still making it accessible for debugging.

'''
