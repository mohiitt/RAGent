import asyncio
from pathlib import Path
import time
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()
st.set_page_config(page_title="RAGent", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
      .app-card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        padding: 1.25rem 1.25rem 1rem;
        background: rgba(255,255,255,0.65);
      }
      .muted {
        color: rgba(0,0,0,0.6);
        font-size: 0.95rem;
      }
      .tight { margin-top: -0.5rem; }
      .spacer { height: 0.5rem; }
      .sources li { margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="ragent_app", is_production=False)

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path

async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="ragent/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )

st.title("📄 RAGent")
st.caption("Upload a PDF to index it, then ask questions across your uploaded documents.")

with st.container():
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.subheader("1) Upload a PDF", anchor=False)
    st.markdown(
        '<p class="muted tight">We’ll store it locally in <code>./uploads</code> and trigger an Inngest event to ingest it.</p>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Only .pdf files are accepted.",
    )

    if uploaded is not None:
        with st.spinner("Uploading and triggering ingestion…"):
            path = save_uploaded_pdf(uploaded)
            # Kick off the event and block until the send completes (unchanged)
            asyncio.run(send_rag_ingest_event(path))
            # Small pause for user feedback continuity (unchanged)
            time.sleep(0.3)

        st.success(f"✅ Triggered ingestion for **{path.name}**")
        st.caption("You can upload another PDF if you like.")
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

st.subheader("2) Ask a question about your PDFs", anchor=False)
st.markdown(
    '<p class="muted tight">We’ll dispatch a query event and poll the local Inngest API until the run finishes.</p>',
    unsafe_allow_html=True,
)

async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="ragent/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]

def _inngest_api_base() -> str:
    # Local dev server default; configurable via env (unchanged)
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)

with st.form("rag_query_form"):
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c1:
        question = st.text_input("Your question", placeholder="e.g., Summarize Section 2 across all PDFs")
    with c2:
        top_k = st.number_input(
            "Top-K",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="How many chunks to retrieve per query.",
        )

    submitted = st.form_submit_button("🔎 Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer…"):
            # Fire-and-forget event to Inngest for observability/workflow (unchanged)
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output (unchanged)
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.markdown("### Answer")
        if answer:
            st.write(answer)
        else:
            st.info("No answer was returned for this query.")

        if sources:
            with st.expander("Sources", expanded=True):
                st.markdown('<ul class="sources">', unsafe_allow_html=True)
                for s in sources:
                    st.markdown(f"<li>{s}</li>", unsafe_allow_html=True)
                st.markdown("</ul>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
st.caption("Built with Streamlit + Inngest • Configure `INNGEST_API_BASE` via environment variable if needed.")
