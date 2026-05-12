import streamlit as st
import os
from rag_engine import RAGEngine

# ── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📄",
    layout="wide"
)

# ── Styles ────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .subtitle {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-top: 1rem;
        color: #000000 !important;
        font-weight: 500;
    }
    .source-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #495057;
        margin-top: 0.5rem;
    }
    .status-badge {
        display: inline-block;
        background: #d4edda;
        color: #155724;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .offline-badge {
        display: inline-block;
        background: #fff3cd;
        color: #856404;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# ── Header ────────────────────────────────────────────
st.markdown('<p class="main-title">📄 Offline PDF Question Answering System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    '<span class="status-badge">✓ Fully Offline</span>'
    '<span class="offline-badge">🔒 Privacy First</span>'
    '&nbsp;&nbsp;Built for secure aerospace & defence environments.'
    '</p>',
    unsafe_allow_html=True
)

# ── Sidebar — Upload ───────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Documents")
    st.caption("Upload one or more PDF files to begin querying.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    chunk_size = st.slider("Chunk Size (tokens)", 200, 800, 400, 50,
                           help="Smaller chunks = more precise answers. Larger = more context.")
    top_k = st.slider("Results to retrieve (Top-K)", 1, 8, 4,
                      help="How many chunks to pull before generating the answer.")

    process_btn = st.button("⚙️ Process Documents", use_container_width=True, type="primary")

    if process_btn and uploaded_files:
        with st.spinner("Reading & indexing documents..."):
            # Save uploads to temp folder
            os.makedirs("temp_docs", exist_ok=True)
            saved_paths = []
            for f in uploaded_files:
                path = f"temp_docs/{f.name}"
                with open(path, "wb") as out:
                    out.write(f.read())
                saved_paths.append(path)

            # Build RAG engine
            groq_api_key = st.session_state.get("groq_api_key", "")
            engine = RAGEngine(chunk_size=chunk_size, top_k=top_k, api_key=groq_api_key)
            engine.load_documents(saved_paths)

            st.session_state.rag = engine
            st.session_state.docs_loaded = True
            st.session_state.chat_history = []

        st.success(f"✅ {len(uploaded_files)} document(s) indexed successfully!")

    if st.session_state.docs_loaded:
        st.markdown("---")
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            st.session_state.rag = None
            st.session_state.docs_loaded = False
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("**Stack**")
    st.caption("LangChain · FAISS · HuggingFace\nStreamlit · PyMuPDF · pdfplumber")

# ── Main Panel ────────────────────────────────────────
if not st.session_state.docs_loaded:
    st.info("👈 Upload PDF documents in the sidebar and click **Process Documents** to begin.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔒 100% Offline")
        st.caption("No data leaves your machine. Built for air-gapped environments.")
    with col2:
        st.markdown("### 📚 Multi-Document")
        st.caption("Query across multiple PDFs simultaneously in one search.")
    with col3:
        st.markdown("### ⚡ Fast Retrieval")
        st.caption("FAISS vector index enables sub-second similarity search.")

else:
    # Chat interface
    st.markdown("### 💬 Ask a Question")

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.success(entry["answer"])
            if entry.get("sources"):
                with st.expander("📎 Source chunks used"):
                    for i, src in enumerate(entry["sources"], 1):
                        st.markdown(
                            f'<div class="source-box"><b>Chunk {i}</b> '
                            f'— {src["source"]}<br><br>{src["content"]}</div>',
                            unsafe_allow_html=True
                        )

    # Input box
    question = st.chat_input("Type your question about the uploaded documents...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = st.session_state.rag.query(question)

            answer = result["answer"]
            if not answer:
                answer = "The model could not generate an answer. Try rephrasing your question."
            st.success(answer)
            if result.get("sources"):
                with st.expander("📎 Source chunks used"):
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(
                            f'<div class="source-box"><b>Chunk {i}</b> '
                            f'— {src["source"]}<br><br>{src["content"]}</div>',
                            unsafe_allow_html=True
                        )

        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", [])
        })
