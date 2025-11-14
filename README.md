# Local RAG Chatbot (FAISS + Sentence-Transformers + Ollama + Streamlit)

This project is a small, fully local Retrieval‑Augmented Generation (RAG) chatbot that:

- Builds semantic embeddings from your data (CSV or SQL) using `sentence-transformers`.
- Stores embeddings in a FAISS index (`data/index.faiss`).
- Stores metadata in MongoDB so the chatbot can fetch the original text.
- Retrieves top‑k context chunks for each user query.
- Sends the context + question to a local LLM via the Ollama Python client.
- Presents a single **Streamlit UI** with a human‑friendly assistant that only answers from your data.

The repository also includes a synthetic demo dataset (`example.csv`) so you can test the flow without real data.

> **Note:** `example.csv` contains synthetic demo records only. It is *not* production data.

---

## Files overview

- `app.py` – main Streamlit app (chat UI + greeting, strict RAG behavior).
- `build_index.py` – builds FAISS index + Parquet metadata from CSV or SQL.
- `database.py` – example script for pulling data from MSSQL.
- `example.csv` – synthetic demo attendance/vehicle data.
- `requirements.txt` – Python dependencies.

---

## 1. Environment setup (CPU‑only and CUDA/GPU)

### 1.1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.2. CPU‑only mode (simplest)

If you don’t have a GPU or CUDA installed, you can run everything on CPU:

- Make sure you have a recent Python (3.9+ recommended).
- Install requirements (done above).
- SentenceTransformers and FAISS will use CPU automatically.

You don’t need to change any code for CPU‑only mode.

### 1.3. Optional: CUDA/GPU acceleration

If you have an NVIDIA GPU and CUDA installed, you can speed up embedding and LLM inference.

1. Install CUDA‑enabled PyTorch (adjust URL/version as needed):

   ```bash
   pip install 'torch>=2.0.0' --index-url https://download.pytorch.org/whl/cu118
   ```

2. Verify GPU is visible:

   ```python
   python - << 'PY'
   import torch
   print("CUDA available:", torch.cuda.is_available())
   PY
   ```

3. SentenceTransformers will automatically use GPU if available. If you want to be explicit in your own scripts when encoding, you can do:

   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")  # or "cpu"
   ```

> The current `build_index.py` and `app.py` instantiate `SentenceTransformer("all-MiniLM-L6-v2")` without an explicit device; if CUDA is visible, it will typically use the GPU.

---

## 2. Ollama setup

The chatbot calls a local LLM via the [Ollama](https://ollama.ai/) Python client.

1. Install Ollama (Linux/macOS):

   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. Pull a model (default in this repo is `llama3`):

   ```bash
   ollama pull llama3
   ```

3. Make sure the Ollama service is running (`ollama` usually runs as a background service).

4. (Optional) Change the model via environment variable:

   ```bash
   export OLLAMA_MODEL=llama3  # or another Ollama model name
   ```

---

## 3. Build the FAISS index

You must build the FAISS index before running the chatbot. There are two common paths:

### 3.1. From the synthetic demo CSV (`example.csv`)

`example.csv` contains a note row and synthetic demo records. To build an index from it:

```bash
python build_index.py --csv example.csv --text-column text --out data
```

> Note: `build_index.py` ignores the note row and constructs concise text summaries from the CSV columns for each record. It then:
>
> - Embeds the summaries with `all-MiniLM-L6-v2`.
> - Writes `data/index.faiss`.
> - Writes `data/metadata.parquet` (id + text).

You can then load these into MongoDB (one document per row) with fields like `faiss_idx`, `id`, and `text`. The chatbot expects that mapping to exist.

### 3.2. From your real data (CSV or SQL)

There are two main options:

#### Option A: Real CSV

1. Export your real data (e.g. attendance/vehicle reports) to a CSV with the same columns as `example.csv` or adapt `load_data_from_csv` in `build_index.py` to your schema.
2. Run:

   ```bash
   python build_index.py --csv your_real_data.csv --out data
   ```

3. Load the resulting metadata (Parquet) into MongoDB and maintain a `faiss_idx` field for each row, matching the FAISS index order.

#### Option B: Directly from SQL (MSSQL / others via pyodbc)

1. Configure a pyodbc connection string.
2. Adjust the SQL query in your own script (or reuse `database.py`) to return an `id` and `text` column or adapt `load_data_from_sql` in `build_index.py` to your table.
3. Run:

   ```bash
   python build_index.py \
       --sql "SELECT id, text FROM your_table" \
       --conn "DRIVER={SQL Server};SERVER=.;DATABASE=db;UID=user;PWD=pwd" \
       --out data
   ```

4. Again, load the generated metadata into MongoDB with a `faiss_idx` field.

---

## 4. MongoDB configuration

The chatbot reads context documents from MongoDB based on FAISS indices. The following environment variables control this:

```bash
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB="vehicle_attendance"
export MONGO_COLLECTION="chatbot_docs"
```

Each document in `chatbot_docs` should have at least:

- `faiss_idx` – integer index position (matching FAISS order).
- `text` – the text chunk for that index.
- Optionally `id` and other metadata.

You can adapt your own loader script to insert these docs based on `data/metadata.parquet` and the FAISS index ordering.

---

## 5. Running the Streamlit chatbot

Once:

- Dependencies are installed.
- Ollama is running with a pulled model.
- FAISS index (`data/index.faiss`) and MongoDB metadata are in place.

Run the app:

```bash
streamlit run app.py --server.port 7860
```

Then open the URL Streamlit prints, e.g. `http://localhost:7860`.

### Chatbot behavior

- Greets the user as “Trashbot assistant”.
- Answers **only** from your indexed company data (via FAISS + MongoDB).
- If the answer is not clearly supported by the retrieved context, it responds with:

  > I don't know the answer based on the company data I have.

- Responses are short, human, and easy to read (1–3 concise sentences or a very short bullet list).

You can tune the strictness via:

```bash
export RAG_STRICT_THRESHOLD=0.35  # higher = more refusals, lower = more answers
```

---

## 6. Troubleshooting

- **`ModuleNotFoundError` for `ollama`, `sentence_transformers`, etc.**
  - Ensure your virtualenv is activated and run `pip install -r requirements.txt`.

- **FAISS index not found**
  - Make sure you ran `build_index.py` and that `data/index.faiss` exists.

- **MongoDB errors**
  - Check that MongoDB is running and credentials/DB/collection names match the environment variables.

- **Slow performance**
  - Consider enabling CUDA (if you have a GPU).
  - Use smaller models or fewer retrieved chunks (tune `TOP_K` in `app.py`).

---

## 7. Extending this project

- Add more structured sources (other SQL tables, APIs) by converting them to text and indexing them.
- Build admin tools to rebuild the index periodically from fresh data.
- Add authentication and role‑based access control around the Streamlit app.
- Swap out the Ollama model for domain‑specific LLMs depending on your use case.

---

## 8. Deployment (Docker, docker-compose, NGINX)

You can containerize and deploy the chatbot using the provided `Dockerfile` and `docker-compose.yml`. A simple architecture looks like:

```
NGINX  →  Streamlit app container (this repo) → Python RAG code + FAISS + MongoDB
          ↑
        Docker

Ollama runs on the host or in a separate Docker container
MongoDB runs as a service (local or Atlas)
```

### 8.1. Build and run with Docker only

```bash
# Build image
docker build -t trashbot-app .

# Run container
docker run --rm -p 7860:7860 \
  -e OLLAMA_MODEL=llama3 \
  -e MONGO_URI="mongodb://host.docker.internal:27017" \
  -e MONGO_DB=vehicle_attendance \
  -e MONGO_COLLECTION=chatbot_docs \
  trashbot-app
```

Then visit `http://localhost:7860`.

> Note: `host.docker.internal` works on macOS/Windows; on Linux you may need to pass the host IP or run MongoDB in Docker (see below).

### 8.2. Run app + MongoDB with docker-compose

The repository includes a `docker-compose.yml` that starts:

- `app` – the Streamlit RAG chatbot (this codebase).
- `mongo` – a MongoDB instance used as metadata store.

Run:

```bash
docker compose up --build
```

This will expose:

- Chatbot at `http://localhost:7860`
- MongoDB at `mongodb://localhost:27017`

You can configure env vars in your shell or an `.env` file (docker-compose will interpolate them), for example:

```bash
export OLLAMA_MODEL=llama3
export EMBEDDER_MODEL=all-MiniLM-L6-v2
export RAG_TOP_K=3
export RAG_CHUNK_CHAR_LIMIT=700
export RAG_STRICT_THRESHOLD=0.35
export RAG_SAFE_MODE=strict
```

### 8.3. Optional: NGINX reverse proxy

For production, you may want to put NGINX in front of the app for TLS and a friendly domain.

- Example config: `deploy/nginx.conf`.
- In `docker-compose.yml`, you can uncomment the `nginx` service and start it alongside the app.

The provided config proxies HTTP traffic on port 80 to the `app` service (Streamlit) inside Docker.

---

## 9. License and contributing

- License: [MIT](LICENSE)
- Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
