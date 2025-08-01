# Bengali RAG System: Technical Journey & Production-Grade Implementation

## 🔍 Executive Overview

This project captures the complete lifecycle of developing a robust multilingual **Bengali Retrieval-Augmented Generation (RAG)** system—from OCR pipelines to dual-vector database architecture and memory-aware interfaces. It chronicles a transition from early-stage experimentation to a **production-level deployment** with intelligent workflows, context awareness, and bilingual evaluation metrics.

---

## 🚀 Development Timeline

### 📄 Phase 1: OCR Foundation
Extract accurate Bengali text from PDF via multi-engine OCR techniques.

### 📦 Phase 2: Vector DB Setup with Pinecone
Implemented scalable semantic search with advanced chunking and embeddings.

### 🔁 Phase 3: Automation via N8N
Added automated classification and routing using low-code workflows.

### 🧠 Phase 4: Memory Management with Supabase
Deployed a dual-database backend with optimized memory-context architecture.

### 🧪 Phase 5: Evaluation + Streamlit UI
Created an interactive, testable, bilingual UI with real-time feedback and download options.

---

## 🛠️ Key Components

### 1. OCR & Text Extraction

- Triple-engine pipeline: `PyMuPDF` → `Tesseract (ben)` → `EasyOCR`
- AI-powered text cleaning via **GPT-4**
- Markdown-ready formatting for RAG integration

### 2. Intelligent Chunking

- Bengali-aware parser with MCQ, narrative, table, and vocabulary detection
- 512-token semantic chunk limits with structural metadata
- ~350 optimized document chunks with content type tagging

### 3. Vector Storage Strategy

- **Pinecone v1**: `bge-m3` embeddings (1024D)
- **Supabase v2**: `text-embedding-3-small` embeddings (1536D)
- Dual-vector schema for MCQ vs narrative chunks

### 4. N8N Automation

- AI classification of content: MCQ vs Narrative
- Dynamic routing to respective DB
- Error isolation, rate-limiting, and metadata enrichment

### 5. Memory-Aware Context Handling

- Session-managed conversation tracking
- Context-injected answers using previous dialogue
- Bilingual pronoun/resolution logic

### 6. Evaluation Framework

- Bengali-specific scoring rubric (accuracy, relevance, clarity)
- Multi-metric scoring: cosine similarity + GPT-based scoring
- Real-time CSV export with weighted scores

### 7. Production Web Interface

- Built on Streamlit
- Features source attribution, debug mode, Supabase status check
- Context-aware input and output with download/export options

---

## 🧠 Architecture Diagram

```
PDF → OCR (3 engines) → AI Cleaning → Chunking
    → Classification (N8N) → Dual Vector DB (MCQ + Narrative)
        → LangChain RAG → Streamlit Chat UI → Evaluation
```

---

## 🏆 Project Highlights

- ✅ First Bengali RAG system with **multi-engine OCR**
- ✅ Real-time, **context-aware question answering**
- ✅ Advanced evaluation pipeline for Bengali educational content
- ✅ Supports **semantic retrieval**, MCQ filtering, and intelligent chunk routing
- ✅ **Open source**, reusable for other Bengali/NLP tasks

---

## 🔮 Roadmap

- 🔊 Bengali speech support
- 📷 Diagram/image question parsing
- 📱 Mobile interface
- 🧪 Query expansion & hybrid dense-sparse retrieval

---

## 📚 Ideal For

- Bengali educational QA systems
- Digital learning & tutoring apps
- Language preservation tools
- Government e-learning systems

---

## 👥 Contributors

- Built by a research-driven team of ML developers, Bengali linguists, and backend engineers.
- Special thanks to contributors who helped implement OCR, LangChain integration, and evaluation modules.

---

## 📂 Repo Structure (Simplified)
```
📁 preprocessing and n8n code/
  ├── 1_upsert_naive_rag.ipynb
  ├── 2_generate_answer_naive_rag.ipynb
📁 pages/
  ├── 05_QAv2.py
  ├── 06_QAv3_memory.py
app.py
Readme.md
```

---

## 💬 Feedback & Support

We welcome contributions and feedback to enhance this system.  
Feel free to raise issues, submit pull requests, or use this system in your own projects!

