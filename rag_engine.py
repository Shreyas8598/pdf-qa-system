"""
rag_engine.py — RAG with Groq LLM (free, fast, no local GPU needed)
"""

import os
import re
from typing import List, Dict, Any

import fitz
import pdfplumber
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


class RAGEngine:

    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GROQ_MODEL      = "llama-3.1-8b-instant"   # free, fast

    def __init__(self, chunk_size: int = 400, top_k: int = 4, api_key: str = ""):
        self.chunk_size  = chunk_size
        self.top_k       = top_k
        self.api_key     = api_key
        self.vectorstore = None
        self._embeddings = None
        self._client     = None

    def _extract_text(self, path: str) -> str:
        try:
            doc  = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception:
            text = ""
        if len(text.strip()) < 100:
            try:
                with pdfplumber.open(path) as pdf:
                    text = "\n".join(
                        p.extract_text() for p in pdf.pages if p.extract_text()
                    )
            except Exception:
                text = ""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def load_documents(self, pdf_paths: List[str]) -> None:
        raw_docs = []
        for path in pdf_paths:
            text = self._extract_text(path)
            if text:
                raw_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": os.path.basename(path)}
                    )
                )
        if not raw_docs:
            raise ValueError("No extractable text found.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size * 0.15),
        )
        chunks = splitter.split_documents(raw_docs)
        self.vectorstore = FAISS.from_documents(chunks, self._get_embeddings())

    def query(self, question: str) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"answer": "No documents loaded yet.", "sources": []}

        docs    = self.vectorstore.similarity_search(question, k=self.top_k)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this information in the document."

Context:
{context}

Question: {question}

Answer:"""

        try:
            client   = self._get_client()
            response = client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"LLM error: {str(e)}"

        sources = [
            {
                "source" : d.metadata.get("source", "unknown"),
                "content": d.page_content[:400] + "..."
                           if len(d.page_content) > 400 else d.page_content
            }
            for d in docs
        ]

        return {"answer": answer, "sources": sources}

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings
