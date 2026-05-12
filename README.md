# 📄 Offline AI-Powered PDF Question Answering System

> A fully offline, privacy-first document QA platform built for aerospace and defence environments where internet connectivity is prohibited.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![Offline](https://img.shields.io/badge/Mode-100%25_Offline-darkgreen?style=flat-square)

---

## 🔍 What It Does

Upload one or more PDF documents and ask natural language questions. The system retrieves relevant chunks from the documents and generates accurate, context-aware answers — entirely offline, with no data leaving your machine.

Built during an internship at **Hindustan Aeronautics Limited (HAL)** to enable secure document querying in air-gapped defence networks.

---

## 🏗️ Architecture

```
PDF Files
    │
    ▼
┌─────────────────────┐
│   PDF Extraction    │  PyMuPDF (primary) → pdfplumber (fallback)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Text Chunking     │  RecursiveCharacterTextSplitter (400 tokens, 15% overlap)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Semantic Embedding │  HuggingFace all-MiniLM-L6-v2 (runs locally)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   FAISS Index       │  Local vector store — no server needed
└─────────┬───────────┘
          │
    User Query
          │
          ▼
┌─────────────────────┐
│  Similarity Search  │  Top-K chunk retrieval
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Answer Generation  │  Flan-T5 (local LLM) via LangChain RetrievalQA
└─────────┬───────────┘
          │
          ▼
     Answer + Sources
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Extraction | PyMuPDF, pdfplumber, PyPDF2 |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (CPU) |
| LLM | Google `flan-t5-base` (local) |
| Orchestration | LangChain RetrievalQA |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-qa-system.git
cd pdf-qa-system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> First run will download the embedding model (~80 MB) and LLM (~250 MB). After that — fully offline.

### 4. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📸 Usage

1. Upload one or more PDF files in the sidebar
2. Adjust chunk size and retrieval count if needed
3. Click **Process Documents**
4. Type your question in the chat box
5. Get answers with source references

---

## 🔒 Privacy & Security

- **Zero network calls** after initial model download
- No data is sent to any external server
- FAISS index is stored in memory — cleared on reset
- Designed for **air-gapped** and **MIL-SPEC** environments

---

## 📁 Project Structure

```
pdf-qa-system/
│
├── app.py              # Streamlit frontend
├── rag_engine.py       # RAG pipeline (extraction → embedding → QA)
├── requirements.txt    # Dependencies
├── README.md           # This file
└── temp_docs/          # Temporary upload storage (auto-created)
```

---

## 🤝 Acknowledgements

Built during internship at **Hindustan Aeronautics Limited (HAL), Korwa** as part of an R&D initiative to enable secure, offline document intelligence for defence applications.

---

## 📬 Contact

**Shreyas Srivastava**  
[LinkedIn](https://linkedin.com/in/shreyas-srivastava) · [GitHub](https://github.com/YOUR_USERNAME)
