# RAG — Hybrid Retrieval-Augmented Generation

An end-to-end **Retrieval-Augmented Generation (RAG)** project built with **LangChain**, **Hugging Face embeddings**, **ChromaDB**, **BM25**, and **Ollama**.

The project combines information from **Wikipedia JSON/JSONL data** and **PDF research papers**, processes the documents using **semantic chunking**, stores their embeddings in ChromaDB, and uses **hybrid retrieval** (dense vector search + BM25 keyword search) before generating answers with a local LLM.

---

## Project Overview

The complete RAG pipeline implemented in this project is:

```text
Wikipedia JSON/JSONL ──┐
                      │
PDF Documents ────────┤
                      ↓
              Document Loading
                      ↓
               Data Preprocessing
                      ↓
              Semantic Chunking
                      ↓
             Document Embeddings
                      ↓
                 ChromaDB
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
 Dense Vector Search           BM25 Search
        │                           │
        └─────────────┬─────────────┘
                      ↓
              Hybrid Retrieval
             (EnsembleRetriever)
                      ↓
              Retrieved Context
                      ↓
                 RAG Prompt
                      ↓
              Ollama / Llama 3.2
                      ↓
                Final Answer
```

---

## Key Features

- Load Wikipedia data from JSON/JSONL files
- Load PDF documents using PyMuPDF
- Clean and convert source data into LangChain `Document` objects
- Semantic chunking using `SemanticChunker`
- Generate embeddings using:
  `sentence-transformers/all-MiniLM-L6-v2`
- Store document embeddings in ChromaDB
- Dense semantic retrieval using ChromaDB
- Sparse keyword retrieval using BM25
- Hybrid retrieval using `EnsembleRetriever`
- Custom RAG prompt with context-grounded answering
- Local LLM inference using Ollama
- Retrieval inspection and testing inside Jupyter Notebook

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG orchestration |
| Hugging Face | Text embeddings |
| Sentence Transformers | `all-MiniLM-L6-v2` embedding model |
| ChromaDB | Vector database |
| BM25 | Sparse keyword retrieval |
| Ollama | Local LLM inference |
| Llama 3.2 | Generation model |
| PyMuPDF | PDF document loading |
| Jupyter Notebook | Development and experimentation |

---

## RAG Pipeline

### 1. Environment Configuration

The project loads environment variables from a `.env` file.

```python
from dotenv import load_dotenv

load_dotenv(".env")
```

The `.env` file can contain configuration such as the path to the Wikipedia JSON/JSONL dataset.

**Do not commit `.env` to GitHub.**

---

### 2. Embedding Model

The project uses the Hugging Face model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It converts text into numerical vectors so that semantically similar content can be retrieved.

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

---

### 3. Wikipedia Data Loading

Wikipedia data is loaded using LangChain's `JSONLoader`.

The raw JSON records are then converted into LangChain `Document` objects containing:

- `title`
- `id`
- `source`
- combined paragraph text

---

### 4. Semantic Chunking

Instead of splitting documents only by a fixed character count, this project uses semantic chunking.

```python
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile"
)
```

The goal is to create chunks based on changes in semantic meaning.

---

### 5. PDF Processing

PDF files are discovered from:

```text
content/*.pdf
```

Each PDF is:

```text
PDF
 ↓
Pages
 ↓
Semantic Chunking
 ↓
Document Chunks
```

The project also contains an alternative fixed-size chunking approach using `RecursiveCharacterTextSplitter`.

---

### 6. ChromaDB Vector Store

All Wikipedia and PDF chunks are combined into one corpus.

```python
total_docs = wiki_chunks + paper_docs
```

The documents are then embedded and stored in ChromaDB.

```python
chroma_db = Chroma.from_documents(
    documents=total_docs,
    embedding=embeddings,
    persist_directory="./my_db",
    collection_metadata={
        "hnsw:space": "cosine"
    }
)
```

The local vector database is stored in:

```text
my_db/
```

For GitHub, this generated database directory should normally be excluded using `.gitignore`.

---

## Hybrid Retrieval

A major part of this project is combining two retrieval strategies.

### Dense Retrieval

ChromaDB performs semantic similarity search using document embeddings.

```python
vector_retriever = chroma_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

This is useful when the query and document use different wording but have similar meaning.

### Sparse Retrieval

BM25 performs keyword-based retrieval.

```python
bm25_retriever = BM25Retriever.from_documents(total_docs)
bm25_retriever.k = 5
```

This is useful when exact or important keywords occur in the documents.

### Hybrid Retrieval

Both retrievers are combined using `EnsembleRetriever`.

```python
hybrid_retriever = EnsembleRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    weights=[0.5, 0.5]
)
```

The current configuration gives equal weight to dense and sparse retrieval.

---

## RAG Prompt

The retrieved documents are passed to a custom prompt.

The prompt instructs the model to:

- Answer using the retrieved context
- Avoid making up information
- Say that it does not know when the answer is not available in the context
- Provide a detailed and well-formatted response
- Include citation/source information when available

This creates the core RAG pattern:

```text
Question + Retrieved Context
            ↓
         RAG Prompt
            ↓
           LLM
            ↓
          Answer
```

---

## Local LLM with Ollama

The generation step uses Ollama with:

```text
llama3.2
```

Configuration:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)
```

Before running the notebook, install Ollama and make sure the selected model is available locally.

Example:

```bash
ollama pull llama3.2
```

Then verify:

```bash
ollama list
```

---

## End-to-End RAG Chain

The complete chain follows:

```text
User Query
    ↓
Hybrid Retriever
    ↓
Relevant Documents
    ↓
Context Formatting
    ↓
RAG Prompt
    ↓
Ollama Llama 3.2
    ↓
Final Answer
```

The notebook constructs the chain using LangChain Runnable components:

```python
qa_rag_chain = (
    {
        "context": (hybrid_retriever | format_docs),
        "question": RunnablePassthrough()
    }
    | rag_prompt_template
    | llm
)
```

---

## Example Query

Example retrieval query:

```text
What is self attention model?
```

Example end-to-end question:

```text
What is Machine Learning?
```

The system retrieves relevant chunks from the combined Wikipedia and PDF corpus and provides the retrieved context to the local LLM.

---

## Project Structure

```text
SimpleRAG/
│
├── content/
│   └── *.pdf
│
├── my_db/                  # Local ChromaDB - do not commit
│
├── .venv/                  # Virtual environment - do not commit
│
├── .env                    # Environment variables - do not commit
│
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── RAG.ipynb
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/simple-rag.git
cd simple-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a local `.env` file.

Example:

```env
JSON_FILE_PATH=path/to/your/wiki_dataset.jsonl
```

Use the actual path required by your dataset.

### 6. Install and run Ollama

Install Ollama, then download the model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running before executing the final RAG chain.

---

## Running the Project

Open:

```text
RAG.ipynb
```

Run the notebook cells in order.

The notebook follows this sequence:

```text
Environment Setup
        ↓
Embeddings
        ↓
Wikipedia Loading
        ↓
Wikipedia Preprocessing
        ↓
Semantic Chunking
        ↓
PDF Loading
        ↓
PDF Chunking
        ↓
Combine Documents
        ↓
ChromaDB
        ↓
Dense + BM25 Retrieval
        ↓
Hybrid Retrieval
        ↓
RAG Prompt
        ↓
Ollama LLM
        ↓
Final Answer
```

---

## Important Notes

### API Keys and Secrets

Never commit:

```text
.env
```

to GitHub.

If your `.env` contains API keys or other credentials, keep it local.

### ChromaDB

The `my_db/` directory is a generated local vector store. It can be recreated by running the vector database creation step.

### Source Documents

The PDF files inside `content/` are used as the project knowledge base. Make sure you have permission to redistribute any documents you publish in a public GitHub repository.

---

## Learning Objectives

This project was built to understand the practical implementation of:

- Retrieval-Augmented Generation
- Embeddings
- Semantic Chunking
- Vector Databases
- Dense Retrieval
- Sparse Retrieval
- BM25
- Hybrid Search
- Context Augmentation
- Prompt Construction
- Local LLM Inference
- LangChain Runnables
- End-to-End RAG Architecture

---

## Future Improvements

Potential next steps for the project include:

- Add a web or Streamlit interface
- Add conversation memory
- Add metadata filtering
- Add reranking
- Add RAG evaluation
- Add retrieval evaluation metrics
- Add source/citation tracking
- Add document ingestion as a reusable pipeline
- Add configurable chunking strategies
- Add LangSmith tracing and observability
- Deploy the application as an API or web application

---

## Author

**Fardin Khan**

Data Science & Generative AI Learner

---

## License

This project can be released under the MIT License if you choose to make it open source.
