# Research Intelligence RAG

An AI-powered research assistant designed to help users explore research papers, retrieve relevant passages, and generate evidence-grounded answers from technical documents.

## Overview

Research Intelligence RAG combines PDF processing, semantic search, hybrid retrieval, reranking, and large language model (LLM) generation in a web application.

Users can upload research papers and ask questions about their content. The system is designed to retrieve relevant passages and use them as context for generating answers with source references.

## Features

- **PDF ingestion:** Extract selectable text from text-based research PDFs using PyMuPDF. Scanned/image-only PDFs are not supported by the current extraction pipeline.
- **Text chunking:** Split extracted text into smaller passages suitable for retrieval.
- **Semantic search:** Generate dense embeddings using BGE.
- **Hybrid retrieval:** Combine dense vector retrieval with BM25 keyword search.
- **Reranking:** Reorder candidate passages to improve retrieval relevance.
- **Grounded answer generation:** Use an LLM to formulate answers from retrieved evidence.
- **Research workspace:** Ask questions, inspect retrieved sources, and review generated answers.
- **Document library:** Upload PDFs and view their registration details.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, Vite |
| Backend API | FastAPI, Python |
| PDF text extraction | PyMuPDF |
| Embeddings | BGE |
| Vector database | Qdrant Cloud |
| Keyword retrieval | BM25 |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | Cross-encoder reranking |
| LLM integration (original plan) | Google Gemini API |
| Relational database | PostgreSQL |
| Frontend deployment | Netlify |
| Backend and worker deployment | Railway |

> **LLM provider note:** Google Gemini was the original provider planned for this project and is displayed in the frontend. Confirm the active backend provider before describing Gemini as the current production provider; the backend was also tested with Groq during development.

## Architecture

The application is organized into a frontend, backend API, and separate ingestion worker.

1. **Document upload:** The frontend sends PDF files to the FastAPI backend.
2. **Text extraction:** PyMuPDF extracts selectable text from text-based PDFs. OCR is not implemented in this project, so scanned/image-only PDFs may not yield usable text.
3. **Chunking and embeddings:** Extracted text is divided into chunks and represented as vectors.
4. **Indexing:** Document vectors are stored in Qdrant for retrieval.
5. **Hybrid retrieval:** Dense search and BM25 retrieve candidate passages, which are combined using rank fusion and reranked.
6. **Answer generation:** Retrieved context is supplied to the configured LLM to generate a response.
7. **Source inspection:** The interface displays retrieved source passages to help users verify answers.

Document registration and successful indexing are separate stages. A successful upload response alone does not confirm that a document has completed ingestion.

## Live Demo

- **Frontend:** https://research-intelligence-rag.netlify.app/
- **Backend health check:** https://research-intelligence-rag-production.up.railway.app/health

The backend health endpoint verifies API availability; it does not guarantee that every ingestion or retrieval operation is working correctly.

## Local Development

### Prerequisites

- Python 3.12
- Node.js and npm
- PostgreSQL connection details
- Qdrant Cloud URL and API key
- An API key for the LLM provider configured in the backend

### Backend setup

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies using the backend dependency file present in the repository. For example, if it contains `backend/requirements.txt`:

```bash
pip install -r backend/requirements.txt
```

Configure the environment variables required by the backend for PostgreSQL, Qdrant, and the selected LLM provider. Use the variable names expected by the application code. Do not commit secrets.

Start the API using the entry point configured in the repository. For the current app layout, the command may be:

```powershell
$env:PYTHONPATH = "backend"
uvicorn app.main:app --reload --port 8000
```

### Frontend setup

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Configure the frontend API base URL to point to the backend you intend to use.

> Setup commands and dependency-file paths can differ by repository version. Check the files in your checkout and use the existing project configuration.

## Configuration and Security

- Keep API keys, database credentials, and other secrets in environment variables.
- Do not commit `.env` files, access tokens, or credentials to GitHub.
- Use a restricted API key where the provider supports it.
- Keep the frontend API base URL aligned with the backend deployment.
- Provider quotas, service availability, and free-tier hosting limits can affect the live demo.

## Current Limitations

- Retrieval quality depends on successful document ingestion, indexing, and passage selection.
- The current ingestion pipeline is designed for text-based PDFs; scanned/image-only PDFs are not supported because OCR is not implemented.
- Generated answers and citations must be checked against the original papers.
- A document marked `registered` may not yet be fully indexed.
- Free-tier hosting, database retention, and API quotas may affect long-term availability.

## Learning Outcomes

This project explores practical implementation of Retrieval-Augmented Generation (RAG), document-processing pipelines, vector search, hybrid information retrieval, reranking, LLM integration, and deployment of a full-stack AI application.

## License

No license is specified yet. Add a license file if you intend to distribute the project under an explicit open-source license.
