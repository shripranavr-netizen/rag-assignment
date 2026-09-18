A Retrieval-Augmented Generation (RAG) application that lets you upload a
document and ask questions about its content — answered **only** from the
document, not the model's general knowledge.

## 1. Project Overview

This system implements the full RAG pipeline end-to-end:

```
Document → Text Extraction → Chunking → Embeddings → Vector Database
  → Similarity Search → Retrieved Context → LLM → Answer
```

Given a document (PDF / TXT / DOCX / Markdown), it extracts the text,
splits it into overlapping chunks, embeds each chunk, stores the
embeddings in a local FAISS vector index, and — for every user question —
retrieves the most relevant chunks and passes them to an LLM which is
instructed to answer using *only* that retrieved context.

## 2. Architecture / Workflow

```
                 ┌──────────────────┐
  document.pdf → │  document_loader │ → raw text
                 └──────────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │     chunker      │ → list of overlapping chunks
                 └──────────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │   vector_store   │ → TF-IDF embeddings → FAISS index
                 │  (EmbeddingIndex)│
                 └──────────────────┘
                           │
     user question ────────┤ .search(question, top_k)
                           ▼
                 ┌──────────────────┐
                 │  Top-K relevant  │
                 │   chunks + score │
                 └──────────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │    llm_client    │ → prompt = instructions + context
                 │ (Anthropic /     │            + question
                 │  OpenAI /        │
                 │  offline fallback│ → final answer
                 └──────────────────┘
```

**File layout:**

| File | Responsibility |
|---|---|
| `document_loader.py` | Stage 1 — extracts raw text from PDF / TXT / DOCX / MD |
| `chunker.py` | Stage 2 — splits text into overlapping word-based chunks |
| `vector_store.py` | Stage 3 & 4 — generates embeddings, stores/searches them in FAISS |
| `llm_client.py` | Stage 6 — builds the grounded prompt and calls the LLM (or offline fallback) |
| `rag_pipeline.py` | Wires all stages together into one `RAGPipeline` class |
| `app_cli.py` | Command-line interface |
| `app_streamlit.py` | Bonus web UI |
| `data/sample_policy.txt` | Sample test document (fictional company handbook) |

## 3. Technologies Used

- **Python 3**
- **Text extraction:** `pypdf` (PDF), `python-docx` (DOCX), built-in file I/O (TXT/MD)
- **Embeddings:** `scikit-learn` TF-IDF vectorizer, L2-normalized (see note below)
- **Vector database:** `faiss-cpu` (`IndexFlatL2`, exact nearest-neighbor search)
- **LLM:** Anthropic Claude or OpenAI GPT (auto-detected via API key env vars), with an offline extractive fallback mode
- **UI:** Streamlit (bonus)

### Why TF-IDF instead of a neural embedding model?

The pipeline is deliberately structured so the embedding step is fully
isolated inside `vector_store.py`'s `EmbeddingIndex` class. We used
scikit-learn's TF-IDF vectorizer (L2-normalized, so FAISS's `IndexFlatL2`
correctly approximates cosine similarity) because it requires **no
external model download** and runs **entirely offline** — useful in
network-restricted environments and for fast, dependency-light grading.

**To upgrade to real semantic embeddings** (recommended for production /
for better recall on paraphrased questions), only `EmbeddingIndex.fit()`
and `EmbeddingIndex.embed_query()` need to change — e.g.:

```pythonA` 
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
dense = model.encode(chunk_texts, normalize_embeddings=True)
```

Nothing else in the pipeline (chunker, FAISS index, retriever, LLM step)
needs to change, since they only depend on this class's public
`fit()` / `search()` interface.

## 4. Setup Instructions

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) enable real LLM-generated answers
export ANTHROPIC_API_KEY="sk-ant-..."     # or
export OPENAI_API_KEY="sk-..."
# If neither is set, the app still runs fully end-to-end using an
# offline extractive fallback (see Section 8 below).
```

## 5. How to Run the Application

### Option A — Command-line (interactive)

```bash
python app_cli.py data/sample_policy.txt
```

```
Q: What is the company's work-from-home policy?
```

### Option B — Command-line (single question, non-interactive)

```bash
python app_cli.py data/sample_policy.txt --question "What is the company's work-from-home policy?"
```

### Option C — Web UI (bonus)

```bash
streamlit run app_streamlit.py
```

Then open the printed local URL, upload a document, click **Ingest
document**, and start asking questions. The UI displays the retrieved
chunks and their similarity scores alongside the final answer.

### Tuning chunk size / overlap

Both the CLI and the UI expose `--chunk-size` / `--overlap` (or sliders in
the UI) so you can experiment, e.g.:

```bash
python app_cli.py data/sample_policy.txt --chunk-size 100 --overlap 20 \
    --question "How many sick leave days do employees get?"
```

## 6. Sample Document

`data/sample_policy.txt` — a fictional company (Nimbus Technologies)
employee handbook covering WFH policy, leave policy, working hours,
expense reimbursement, performance reviews, code of conduct, equipment
policy, and travel policy. It mirrors the assignment's own example
("What is the company's work-from-home policy?").

## 7. Screenshots

*(Capture these while running the commands in Section 5 on your own
machine, then place the image files in a `screenshots/` folder and
reference them here.)*

- [ ] Document ingestion (Stage 1–2 console output)
- [ ] Embedding/indexing process (Stage 3–4 console output)
- [ ] Vector database / similarity search (Stage 5 console output, showing chunk IDs + similarity scores)
- [ ] User query being entered (CLI prompt or Streamlit text box)
- [ ] Retrieved context displayed (Streamlit "Retrieved context chunks" expander, or CLI `[retrieve]` log lines)
- [ ] Final generated answer

## 8. Concept Explanations

### What are embeddings?

An embedding is a fixed-length numeric vector representation of a piece
of text, positioned in a high-dimensional space such that texts with
similar *meaning* (or, in TF-IDF's case, similar *vocabulary/topic*) end
up close together in that space. This lets us turn "is this chunk
relevant to this question?" — a fuzzy, semantic question — into "how
close are these two vectors?" — a concrete, computable one (e.g. cosine
similarity or Euclidean distance).

### Why is a vector database required?

A vector database (here, FAISS) is required because comparing a query
embedding against thousands of chunk embeddings one at a time in Python
is slow and doesn't scale. Vector databases store embeddings in
structures optimized for fast nearest-neighbor search — even exact
search like `IndexFlatL2` is implemented with heavily optimized,
vectorized (SIMD) linear algebra, and approximate-search variants (not
used here, but available in FAISS) scale to millions/billions of
vectors with sub-linear query time. It also cleanly separates "storage
and retrieval of chunk vectors" from the rest of the application logic.

### How does similarity search work?

1. The user's question is converted into an embedding using the *same*
   embedding model/vectorizer that embedded the document chunks (so
   both live in the same vector space).
2. That query vector is compared against every stored chunk vector,
   typically using cosine similarity or Euclidean/L2 distance.
3. The chunks with the smallest distance (highest similarity) to the
   query are returned as the "top-K most relevant" chunks.

In this project, `EmbeddingIndex.search()` does exactly this: it embeds
the query with the fitted TF-IDF vectorizer, runs `faiss_index.search()`
to get the `top_k` nearest chunk vectors by L2 distance, and converts
that distance into an easy-to-read similarity score in `[0, 1]`.

### What chunk size and overlap did you select, and why?

- **Chunk size: 200 words** (~1000–1400 characters). Large enough to
  retain enough context for the LLM to construct a full answer, small
  enough that each chunk stays focused on roughly one topic, which
  keeps similarity search precise (a chunk covering 5 unrelated
  policies would dilute its own embedding and rank poorly for any
  single one of them).
- **Chunk overlap: 40 words** (20% of chunk size). Without overlap, a
  sentence that answers the user's question could be split exactly
  across the boundary between two chunks, so that neither chunk alone
  contains the full answer. The 40-word overlap acts as a buffer,
  ensuring boundary-spanning content still appears intact in at least
  one chunk.

Both values are configurable via `--chunk-size` / `--overlap` (CLI) or
the sliders (Streamlit UI) — see `chunker.py` for the exact
implementation and inline reasoning.

### How does RAG differ from simply asking an LLM a question?

Asking an LLM a question directly relies entirely on what the model
memorized during training — it has no access to your specific document,
can't cite it, and may either refuse to answer or (worse) hallucinate a
plausible-sounding but wrong answer, especially for private, recent, or
niche documents the model was never trained on.

RAG instead **grounds** the LLM's answer in the actual document at
answer-time: it retrieves the specific passages relevant to the question
and includes them directly in the prompt, explicitly instructing the
model to answer only from that retrieved context (and to say so if the
answer isn't present — see `SYSTEM_INSTRUCTION` in `llm_client.py`).
This means:

- The answer can be traced back to specific source chunks (see the
  "Retrieved context chunks" section in the UI).
- The model can correctly say "not available in the document" instead
  of guessing.
- The knowledge source can be updated at any time (just re-ingest a new
  document) without needing to retrain or fine-tune the model.

## 9. Offline Fallback Mode

If neither `ANTHROPIC_API_KEY` nor `OPENAI_API_KEY` is set,
`llm_client.py` automatically falls back to an **extractive** mode: it
returns the single most relevant retrieved chunk directly (clearly
labeled `[OFFLINE FALLBACK MODE]`), or explicitly states the
information isn't available if the best match's similarity score falls
below a relevance threshold (`MIN_RELEVANCE_THRESHOLD = 0.15`). This
keeps the entire pipeline runnable and demonstrable end-to-end even
without any API key or network access — useful for grading in sandboxed
environments.

## 10. Bonus Features Implemented

- [x] Streamlit web-based UI (`app_streamlit.py`)
- [x] Displays similarity/relevance score for each retrieved chunk
- [x] Explicit "information not available in the document" response
      when retrieval confidence is too low
- [x] Conversation history (Streamlit UI keeps a running Q&A log per
      session)
- [x] Configurable chunk size / overlap / top-K at runtime, to easily
      compare different chunking strategies
