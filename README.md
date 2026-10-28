# TaskPilot AI

TaskPilot AI is a personal AI agent MVP built with Python, FastAPI, Streamlit, Gemini function calling, ChromaDB, PyMuPDF, Pandas, Plotly, and SQLite.

## Features

- Ollama-powered agent with a bounded tool-calling loop.
- Safe arithmetic calculator, note saving, CSV analysis, document search, and report creation tools.
- PDF/TXT ingestion with chunking, Gemini embeddings, persistent ChromaDB retrieval, and filename/page references.
- CSV data-quality summaries, previews, numeric statistics, and interactive charts.
- Markdown/PDF report downloads.
- Local SQLite task history and notes.
- Streamlit dashboard with tool execution transparency.

## Requirements

- Python 3.11 recommended (3.10+ may work).
- A Gemini API key from Google AI Studio.
- Internet access for Gemini API calls and initial package installation.

## Windows PowerShell setup

```powershell
# Open this folder in VS Code, then run:
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY=your_actual_key`. Never commit `.env`.

Run the Streamlit dashboard:

```powershell
python -m streamlit run frontend/app.py
```

Run the FastAPI backend in a separate terminal:

```powershell
python -m uvicorn backend.main:app --reload
```

- Dashboard: http://localhost:8501
- API health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs

## Quick demo

1. Open **CSV Analysis** and upload `data/sample_sales.csv`.
2. Build a bar chart with `month` as X and `sales` as Y.
3. Open **Documents (RAG)**, upload a text-based PDF or TXT file, and index it.
4. Open **Agent Chat** and ask a question about the document or request CSV summary.
5. Use **Reports & History** to generate a downloadable report.

## Important limitations

- This is a local MVP, not an unrestricted autonomous computer agent.
- The agent can only use its explicitly registered tools. It does not execute arbitrary shell commands or arbitrary Python.
- Document embedding and chat require a valid Gemini API key and are subject to provider quotas.
- Scanned/image-only PDFs need OCR, which is not included.
- The FastAPI service exposes `/health`, `/chat`, and `/tasks`. The Streamlit app calls agent services directly so it can run without the API process; run FastAPI separately to demonstrate the REST API.
- Test all flows locally before listing the project as complete on a resume.

## Tests

```powershell
python -m pytest -q
```

## Project layout

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

## Security notes

- Keep API keys in `.env`; do not paste them into source code or commit them.
- Uploaded files are stored locally under `data/uploads`.
- Tools use allowlisted operations; do not add unrestricted shell or code execution without a strong sandbox.
- This demo is intended for local use. Add authentication, request limits, and deployment hardening before exposing it publicly.
