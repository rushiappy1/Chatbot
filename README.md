# Local RAG Chatbot (FAISS + Sentence-Transformers + Ollama)

This small prototype shows how to build a fully local Retrieval-Augmented Generation (RAG) chatbot that:

- Builds semantic embeddings from your data (CSV or SQL via pyodbc)
- Stores embeddings in FAISS
- Retrieves top-k context chunks for a user query
- Sends the context + question to a local LLM via the Ollama CLI
- Presents the answer in a Gradio chat UI

Files created
- `requirements.txt` - Python dependencies
- `build_index.py` - Build FAISS index and metadata from CSV or SQL
- `chat_ui.py` - Gradio UI that queries FAISS and Ollama

Quick start

1. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Build an index from `example.csv` (demo synthetic data)

```bash
python build_index.py --csv example.csv --text-column text --out data
```

3. Install Ollama and pull a model (optional but recommended)

Follow instructions at https://ollama.ai/

```bash
# install via their script (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3
```

4. Run the Gradio chat UI

```bash
python chat_ui.py
```

Notes & troubleshooting

- If you don't have Ollama, replace `call_ollama` in `chat_ui.py` with a local LLM runner or a simple echo function for testing.
- The chunking strategy in `build_index.py` is naive; consider using a text splitter that respects sentence boundaries and token counts.
- For production with many documents, use an on-disk FAISS index and tune index type.

Next steps (optional)

- Add conversational memory (store previous Q/A and include in prompt)
- Add reranking or a stronger embedder for higher accuracy
- Add a FastAPI backend and authentication if exposing over network
