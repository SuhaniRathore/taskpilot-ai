import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from backend.config import APP_NAME, UPLOAD_DIR, REPORT_DIR, MAX_UPLOAD_MB
from backend.database import init_db, get_recent_tasks, get_notes
from backend.services.agent import run_agent
from backend.services.rag import index_document, list_indexed_documents, delete_document
from backend.services.data_analysis import analyze_csv, build_chart
from backend.services.reports import create_pdf_report, create_markdown_report

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")
init_db()

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#172554,#312e81,#164e63); color: white; margin-bottom: 1rem;}
.hero h1 {margin:0; font-size:2rem;} .hero p {opacity:.85; margin:.4rem 0 0 0;}
.small-muted {color: #8b8b8b; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧭 TaskPilot AI")
    st.caption("Your personal tool-using AI agent")
    page = st.radio("Workspace", ["Agent Chat", "Documents (RAG)", "CSV Analysis", "Reports & History"])
    st.divider()
    st.caption("Ollama-powered • Local SQLite • ChromaDB")
    if st.button("Clear current chat"):
        st.session_state.messages = []
        st.rerun()

st.markdown('<div class="hero"><h1>TaskPilot AI</h1><p>Turn natural-language requests into useful, traceable actions.</p></div>', unsafe_allow_html=True)

if page == "Agent Chat":
    st.subheader("Talk to your agent")
    st.caption("Try: “Calculate 18% of 245”, “Summarize my uploaded PDF”, or “Analyze sales.csv”.")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tools"):
                with st.expander("Tool execution details"):
                    for item in msg.get("tool_results", []):
                        st.write(f"**{item['name']}**")
                        st.json(item.get("result"))
    prompt = st.chat_input("What would you like TaskPilot to do?")
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-8:]]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Planning and using tools..."):
                result = run_agent(prompt, history)
            st.markdown(result["answer"])
            if result.get("tools_used"):
                with st.expander("Tool execution details", expanded=True):
                    for item in result.get("tool_results", []):
                        st.write(f"**{item['name']}**")
                        st.json(item.get("result"))
        st.session_state.messages.append({
            "role": "assistant", "content": result["answer"],
            "tools": result.get("tools_used", []), "tool_results": result.get("tool_results", [])
        })

elif page == "Documents (RAG)":
    st.subheader("Document knowledge base")
    st.write("Upload a text-based PDF or TXT file, then ask questions about its contents in Agent Chat.")
    uploaded = st.file_uploader("Upload document", type=["pdf", "txt"], key="document_upload")
    if uploaded:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
        elif st.button("Index document", type="primary"):
            safe_name = Path(uploaded.name).name
            destination = UPLOAD_DIR / safe_name
            destination.write_bytes(uploaded.getvalue())
            try:
                with st.spinner("Extracting, embedding, and indexing chunks..."):
                    info = index_document(destination, safe_name)
                st.success(f"Indexed {info['chunks_indexed']} chunks from {info['pages_with_text']} pages.")
                st.json(info)
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
    st.markdown("### Indexed documents")
    docs = list_indexed_documents()
    if not docs:
        st.info("No indexed documents yet.")
    for doc in docs:
        c1, c2 = st.columns([4, 1])
        c1.write(doc["filename"])
        if c2.button("Delete", key="delete_" + doc["filename"]):
            delete_document(doc["filename"])
            st.rerun()

elif page == "CSV Analysis":
    st.subheader("Explore a CSV file")
    st.write("Upload a CSV to inspect data quality, numeric summaries, and charts.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
    if uploaded:
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
        else:
            destination = UPLOAD_DIR / Path(uploaded.name).name
            destination.write_bytes(uploaded.getvalue())
            st.success(f"Saved {destination.name}")
    csv_files = sorted(p.name for p in UPLOAD_DIR.glob("*.csv") if p.is_file())
    if csv_files:
        filename = st.selectbox("Choose a file", csv_files)
        try:
            summary = analyze_csv(filename)
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", summary["rows"])
            col2.metric("Columns", summary["columns_count"])
            col3.metric("Missing cells", sum(summary["missing_values"].values()))
            df = pd.read_csv(UPLOAD_DIR / filename)
            with st.expander("Preview data", expanded=True):
                st.dataframe(df.head(100), use_container_width=True)
            with st.expander("Missing values"):
                st.json(summary["missing_values"])
            with st.expander("Numeric summary"):
                st.dataframe(df.describe(include="number").T, use_container_width=True)
            st.markdown("### Chart builder")
            x_col = st.selectbox("X-axis / category", [str(c) for c in df.columns])
            numeric_cols = [str(c) for c in df.select_dtypes(include="number").columns]
            y_options = ["(count only)"] + numeric_cols
            y_choice = st.selectbox("Y-axis (optional)", y_options)
            chart_type = st.selectbox("Chart type", ["bar", "line", "scatter", "histogram"])
            if st.button("Build chart"):
                y_col = None if y_choice == "(count only)" else y_choice
                try:
                    fig = build_chart(filename, x_col, y_col, chart_type)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as exc:
                    st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not analyze CSV: {exc}")
    else:
        st.info("Upload a CSV to begin.")

elif page == "Reports & History":
    st.subheader("Create a report")
    title = st.text_input("Report title", "TaskPilot AI Summary")
    content = st.text_area("Report content", height=220, placeholder="Paste findings, analysis, or notes here...")
    c1, c2 = st.columns(2)
    if c1.button("Generate Markdown report", type="primary"):
        report = create_markdown_report(title, content)
        st.success(f"Created {report['filename']}")
        st.download_button("Download Markdown", report["content"], file_name=report["filename"], mime="text/markdown")
    if c2.button("Generate PDF report"):
        try:
            pdf_path = create_pdf_report(title, content)
            st.success(f"Created {pdf_path.name}")
            st.download_button("Download PDF", pdf_path.read_bytes(), file_name=pdf_path.name, mime="application/pdf")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")
    st.divider()
    st.subheader("Recent agent tasks")
    tasks = get_recent_tasks(50)
    if tasks:
        for task in tasks:
            with st.expander(f"#{task['id']} · {task['created_at']} · {task['prompt'][:80]}"):
                st.markdown("**Request**")
                st.write(task["prompt"])
                st.markdown("**Response**")
                st.write(task["response"])
                if task["tool_names"]:
                    st.caption("Tools: " + task["tool_names"])
    else:
        st.info("No saved tasks yet.")
    st.divider()
    st.subheader("Saved notes")
    notes = get_notes()
    if notes:
        for note in notes:
            with st.expander(note["title"]):
                st.write(note["content"])
                st.caption(note["created_at"])
    else:
        st.caption("Notes created by the agent will appear here.")
