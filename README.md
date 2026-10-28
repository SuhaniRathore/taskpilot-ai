# TaskPilot AI

TaskPilot AI is a local AI productivity assistant built with Python, FastAPI, Streamlit, Ollama, ChromaDB, PyMuPDF, Pandas, Plotly, and SQLite.

## Features

- Local AI chat using Ollama with a bounded tool-calling loop.
- Safe arithmetic calculator, note saving, CSV analysis, document search, and report creation.
- PDF/TXT ingestion, text chunking, local embeddings, and persistent ChromaDB retrieval with filename/page references.
- CSV data-quality summaries, previews, numeric statistics, and interactive charts.
- Markdown and PDF report generation.
- Local SQLite task history and notes.
- Streamlit dashboard with tool execution transparency.
- FastAPI endpoints for health checks, chat, and tasks.

## Technology Stack

- Python 3.11
- FastAPI and Uvicorn
- Streamlit
- Ollama: `qwen2.5:1.5b` for chat
- Ollama: `nomic-embed-text` for embeddings
- ChromaDB and SQLite
- Pandas and Plotly
- PyMuPDF and ReportLab

## Requirements

- Python 3.11 recommended.
- Ollama installed and running.
- The required Ollama models downloaded locally.
- Internet access for initial package installation.

The default configuration uses local Ollama, so a Gemini API key is not required.

## Windows PowerShell Setup

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama from https://ollama.com and download the models:

```powershell
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

Create the environment file:

```powershell
Copy-Item .env.example .env
```

Check that `.env` contains the following local configuration:

```dotenv
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:1.5b
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Keep `.env` private and never commit API keys or secrets.

## Run the Application

Start the Streamlit dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

In a separate terminal, start the FastAPI backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

- Dashboard: http://localhost:8501
- API health: http://127.0.0.1:8000/health
- API documentation: http://127.0.0.1:8000/docs

The Streamlit dashboard calls agent services directly, so it can run without the FastAPI process. Run FastAPI separately to demonstrate the REST API.

## Quick Demo

1. Open **Agent Chat** and test a calculation.
2. Open **Documents (RAG)**, upload a text-based PDF or TXT file, and index it.
3. Ask a question about the indexed document.
4. Open **CSV Analysis** and use `data/sample_sales.csv`.
5. Build a chart with `month` as X and `sales` as Y.
6. Open **Reports & History** and generate a Markdown or PDF report.

## Tests

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
```

The current test suite has 28 tests. Run the tests locally to verify your setup.

## Important Limitations

- This is a local MVP, not an unrestricted autonomous agent.
- Chat and document embeddings require the configured Ollama models to be available locally.
- Performance depends on available RAM and CPU.
- Scanned or image-only PDFs require OCR, which is not included.
- An optional Gemini provider remains available in the code and may require a valid API key and be subject to provider quotas.
- Retrieval quality depends on document content, chunking, embeddings, and the question asked.

## Security

- Never commit `.env`, API keys, passwords, or private documents.
- Uploaded files, local databases, and vector indexes should remain local unless intentionally shared.
- Add authentication, request limits, and deployment hardening before exposing the application publicly.

## Project Structure

```text
backend/
  main.py
  config.py
  database.py
  services/
    agent.py
    rag.py
    data_analysis.py
    reports.py
frontend/
  app.py
tests/
  test_core.py
data/
  sample_sales.csv
```

## License

No license has been added yet. Add one if you intend to grant others permission to reuse or distribute this project.