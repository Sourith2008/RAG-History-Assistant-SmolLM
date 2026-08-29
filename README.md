# 📚 Class IX History RAG Assistant — SmolLM Edition

A lightweight Retrieval-Augmented Generation (RAG) chatbot that answers questions from the **NCERT Class IX History textbook**, *India and the Contemporary World – I*, grounded strictly in the source material rather than the model's general knowledge.

This is a **CPU-optimized, redeployed version** of the original [RAG-History-Assistant](https://github.com/Sourith2008/RAG-History-Assistant) project, rebuilt after the original deployment failed due to hosting resource constraints.

🔗 **Live Demo:** [rag-history-assistant-smollm-fq34u3zuatfegbznubmueh.streamlit.app](https://rag-history-assistant-smollm-fq34u3zuatfegbznubmueh.streamlit.app/)
💻 **Original Project:** [Sourith2008/RAG-History-Assistant](https://github.com/Sourith2008/RAG-History-Assistant)

---

## Note on the Deployment Version

The original version of this assistant used CrossEncoder reranking and Qwen2.5-0.5B-Instruct for answer generation. Due to CPU and deployment constraints, the CrossEncoder reranking stage was removed and Qwen2.5-0.5B-Instruct was replaced with the lightweight **SmolLM2-135M-Instruct** model. The current deployed version therefore uses a simplified retrieval pipeline optimized for lightweight CPU-based deployment.

**Original:**
```
Question → Embedding → FAISS Retrieval → CrossEncoder Reranking → Qwen2.5 → Answer
```

**Current deployed version:**
```
Question → Embedding → FAISS Retrieval → SmolLM2-135M-Instruct → Answer
```

## Overview

The system converts a user's question into a semantic embedding, retrieves the most relevant textbook passages using FAISS, and passes that context directly to SmolLM2-135M-Instruct, which generates a concise, context-grounded answer. The reranking stage present in the original project was dropped to keep the pipeline fast and memory-light enough to run reliably on free-tier CPU hosting.

## Features

- 🔎 Semantic search over the full NCERT Class IX History (Book I) textbook
- 🪶 Single-stage retrieval (FAISS only) for a minimal, fast, low-memory pipeline
- 🤖 Context-grounded generation with a compact instruction-tuned LLM (SmolLM2-135M-Instruct)
- 💬 Chat interface with conversation history, example questions, and a "Clear Chat" control
- ⚡ Built to run reliably within free-tier CPU deployment limits (e.g., Streamlit Community Cloud)

## Tech Stack

| Component | Technology |
|---|---|
| UI / App framework | [Streamlit](https://streamlit.io/) |
| Embedding model | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Vector index | [FAISS](https://github.com/facebookresearch/faiss) |
| Generation model | [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) |
| ML framework | PyTorch / Hugging Face Transformers / Accelerate |

## Project Structure

```
RAG-History-Assistant-SmolLM/
├── app/
│   ├── model.py         # Loads and caches the embedding model + SmolLM2-135M-Instruct
│   └── inference.py      # Retrieval + generation pipeline
├── data/
│   ├── NCERT_CLASS_9_HISTORY_INDIA_AND_CONTEMPORARY_WORLD_1.pdf   # Source textbook
│   ├── docs.pkl           # Chunked textbook passages
│   └── index.faiss        # Prebuilt FAISS vector index
├── streamlit_app.py       # Streamlit chat UI
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sourith2008/RAG-History-Assistant-SmolLM.git
   cd RAG-History-Assistant-SmolLM
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run streamlit_app.py
   ```

The app will be available at `http://localhost:8501`. On first run, the embedding and generation models will be downloaded from Hugging Face and cached locally.

## How It Works

- **`app/model.py`** loads and `@st.cache_resource`-caches the SmolLM2-135M-Instruct model, tokenizer, and the Sentence Transformer embedding model so they're only initialized once per session.
- **`app/inference.py`** loads the prebuilt FAISS index and passage store (`docs.pkl`) at import time, embeds the user's query, retrieves the top-5 nearest passages, and passes the top 2 directly as context to SmolLM2 (no reranking stage). A constrained system prompt instructs the model to answer in one concise sentence using only the given context, and to explicitly say when the answer isn't available in the text.
- **`streamlit_app.py`** renders the chat UI, maintains conversation history in session state, and appends a lightweight instruction hint to the query based on simple keyword matching (e.g., "who is", "what is") to nudge the model toward one-sentence answers for direct-fact questions.

## Limitations

- ⚠️ **Answer depth:** Because SmolLM2-135M-Instruct is a very small (135M-parameter) language model, and generation is capped at **80 new tokens**, this assistant is **not suited for essay-type, multi-paragraph, or highly detailed answers**. It may struggle with nuanced, multi-part, or open-ended questions, and answers can occasionally be factually imprecise, incomplete, or oddly phrased.
- The reranking stage from the original pipeline was removed for performance reasons, so retrieved context is not re-ranked for relevance — this can occasionally surface less relevant passages compared to the original Qwen2.5 + CrossEncoder version.
- Answers are only as good as the retrieved context — ambiguous or out-of-scope questions may return incomplete or "not available in the provided text" responses.
- Currently scoped to a single textbook (*India and the Contemporary World – I*); it will not answer questions outside this source.
- Best suited for short, direct, fact-based questions ("Who was X?", "What is Y?") rather than long-form explanations or essay-style answers.

## License

See [LICENSE](./LICENSE) for details.

## Acknowledgements

- [NCERT](https://ncert.nic.in/) for the source textbook content
- [Hugging Face](https://huggingface.co/) for SmolLM2, Sentence Transformers, and the Transformers library
- [Facebook AI Research](https://github.com/facebookresearch/faiss) for FAISS
- [Streamlit](https://streamlit.io/) for the app framework and free-tier hosting
