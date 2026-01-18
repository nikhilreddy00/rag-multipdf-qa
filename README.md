# Notebook RAG: Ask Questions About a PDF (LangChain + DocArray + Groq)

This project demonstrates a complete **Retrieval-Augmented Generation (RAG)** workflow inside a Jupyter Notebook (`notebook.ipynb`). It enables **question answering over a PDF** by:
1) loading the PDF,  
2) chunking the text,  
3) embedding the chunks,  
4) storing them in a vector store,  
5) retrieving relevant context for a user question, and  
6) generating a grounded answer with a **Groq-hosted LLM** through **LangChain**.

The example PDF used in the notebook is *"Attention Is All You Need"* (`aiayn.pdf`).  
You can replace it with any PDF and reuse the same pipeline. :contentReference[oaicite:2]{index=2}

---

## What you will learn from this notebook

By running `notebook.ipynb`, you will learn how to:

- Connect to an LLM (Groq) using `langchain-groq`
- Build a simple prompt + output parser chain (LCEL)
- Load PDF documents using `PyPDFLoader`
- Split documents using `RecursiveCharacterTextSplitter`
- Generate embeddings using `sentence-transformers`
- Store and search embeddings using `DocArrayInMemorySearch`
- Convert a vector store into a retriever and test retrieval quality
- Create a RAG chain using LangChain Expression Language (LCEL)
- Stream responses token-by-token
- Run multiple questions in parallel (batch mode)

---

## RAG Pipeline (High-level)

### 1) Load environment variables
The notebook reads your Groq API key using `python-dotenv`.

**Why?**  
You should never hardcode secrets (API keys) in code or notebooks.

---

### 2) Initialize Groq LLM
The notebook uses a Groq model (e.g., a Llama variant) via `ChatGroq`.

**Why Groq?**
Groq is optimized for low-latency inference, so answers appear quickly.

---

### 3) Load PDF -> Documents
The notebook loads the PDF using:

- `PyPDFLoader` (from `langchain_community.document_loaders`)

This produces LangChain `Document` objects containing:
- `page_content` (the extracted text)
- `metadata` (page number, source file, etc.)

---

### 4) Split documents into chunks
The notebook then uses `RecursiveCharacterTextSplitter` to chunk long text.

**Why chunking is required**
LLMs have context limits. Chunking allows you to:
- store smaller pieces for retrieval
- retrieve only what’s relevant
- reduce hallucinations (answers stay grounded)

---

### 5) Create embeddings
The notebook uses **SentenceTransformers** embeddings (e.g., MiniLM) to convert each chunk into a dense vector.

**Why embeddings?**
Embeddings represent the *meaning* of text. This allows semantic search:
- A question like “What is d_model?” retrieves chunks that explain it even if the exact phrasing differs.

---

### 6) Store vectors in a vector database
The notebook uses:

- `DocArrayInMemorySearch` (in-memory vector store)

**Why a vector store?**
It supports fast similarity search over embeddings.

> Note: Your `requirements.txt` also includes FAISS and ChromaDB, so you can extend this project to FAISS/Chroma persistence later. :contentReference[oaicite:3]{index=3}

---

### 7) Retrieve context for a question
The notebook converts the vector store into a retriever:

- `vectorstore.as_retriever()`

Then the retriever finds the top relevant chunks for the question.

---

### 8) Generate grounded answers (RAG)
The notebook builds a prompt like:

- “Answer the question based on the context below. If you can't answer, say ‘I don’t know’.”

**Why this prompt?**
It forces the model to stay within retrieved text, improving faithfulness.

Then it builds an LCEL chain that:
- takes a question
- retrieves context
- formats prompt
- calls the LLM
- parses output to a string

---

## Key Notebook Capabilities

### ✅ Single question answering
Ask one question and get one answer grounded in the PDF.

### ✅ Streaming output
The notebook demonstrates streaming responses (token-by-token), useful for UI apps.

### ✅ Batch processing (multiple questions)
The notebook runs a list of questions in parallel using `.batch()`:
- useful for evaluation
- useful for building automated Q&A pipelines

---

## Tech Stack

Core dependencies used in this project include: LangChain (core + community), PDF loaders (PyPDF/PyMuPDF), sentence-transformers embeddings, FAISS/Chroma packages (available for upgrades), Groq integration, dotenv, and docarray. :contentReference[oaicite:4]{index=4}

---

## Setup Instructions

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
