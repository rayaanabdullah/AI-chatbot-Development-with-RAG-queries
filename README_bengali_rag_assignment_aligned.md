# Bengali RAG System – AI-Powered Multilingual Document QA API

## 🎯 Objective

Develop a domain-specific, Retrieval-Augmented Generation (RAG) system that processes **Bengali educational PDFs** and enables intelligent **question-answering** through a streamlined pipeline consisting of OCR, chunking, embedding, vector storage, and a conversational UI.

This system focuses on **Bengali text** and supports scalable document ingestion and answer generation using OpenAI/HuggingFace models, with plans for broader multimodal support.

---

## 🧠 Core Capabilities

- 📄 **Document Types Supported**:
  - `.pdf` (searchable & scanned)
  - `.txt`, `.md` files
  - `.jpg`, `.png` via OCR (EasyOCR, Tesseract)
- 📌 Note: `.docx`, `.csv`, `.db` not currently supported but easily extendable.

- 🧾 **Pipeline Summary**:
  - **OCR**: PyMuPDF, EasyOCR, and Tesseract
  - **Embedding**: OpenAI (`text-embedding-3-small`) or `BAAI/bge-m3`
  - **Vector Store**: Pinecone and Supabase (dual-db)
  - **UI**: Streamlit app with conversation history and source highlighting
  - **APIs**: FastAPI integrated for backend access

---

## 🗂️ Folder Structure

```
📁 preprocessing and n8n code/
  ├── 1_upsert_naive_rag.ipynb  # Preprocessing & Vector DB Ingestion
  ├── 2_generate_answer_naive_rag.ipynb  # QA testing
📁 pages/
  ├── 05_QAv2.py  # Streamlit interface
  ├── 06_QAv3_memory.py  # Memory-enhanced Streamlit interface
app.py  # FastAPI backend
Readme.md
```

---

## 🚀 Setup Instructions

### ✅ 1. Clone & Setup Environment

```bash
git clone <repo>
cd RAG_banglabook-main
python -m venv venv
venv/Scripts/activate  # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

---

### ✅ 2. Preprocess Documents

Run notebook:
```bash
jupyter notebook preprocessing and n8n code/1_upsert_naive_rag.ipynb
```
This will:
- Run OCR
- Clean & chunk documents
- Embed & upload to vector DB

---

### ✅ 3. Launch Application

For Streamlit UI:
```bash
streamlit run pages/06_QAv3_memory.py
```

For API:
```bash
uvicorn app:app --reload
```

Then visit:
- `http://127.0.0.1:8000/docs` for API docs
- `http://localhost:8501` for chat interface

---

## 📡 API Usage (Partial Implementation)

### POST /query (via FastAPI)
```json
{
  "question": "কল্যাণীর প্রকৃত বয়স কত ছিল?",
  "file_id": "bangla_pdf_01",
  "image_base64": null
}
```

### Response
```json
{
  "answer": "১৫ বছর",
  "sources": ["Page 03 - bangla_book.pdf"]
}
```

---

## 🔍 Key Features

| Feature | Description |
|--------|-------------|
| ✅ Multi-engine OCR | PyMuPDF + EasyOCR + Tesseract |
| ✅ Bengali-aware chunking | Custom parser for MCQ, narrative, tables |
| ✅ Dual vector DB | Pinecone + Supabase, type-routed |
| ✅ Context-aware QA | Maintains conversation memory |
| ✅ Evaluation framework | Custom Bengali + cosine similarity |
| ✅ Streamlit UI | Real-time QA + debugging tools |

---

## 🔐 Sample `.env`
```
OPENAI_API_KEY=your-key
SUPABASE_URL=https://xyz.supabase.co
SUPABASE_KEY=your-key
```

---

## 🌱 Future Work

- 🧠 Add `.docx`, `.csv`, `.db` support
- 🖼️ Integrate vision models for diagram questions
- 📱 Mobile-first UI
- 🔁 Multi-document querying
- 🐳 Docker support for full deployment

---

## 🙌 Acknowledgment

This project is tailored for Bengali NLP tasks but follows the architecture of general-purpose RAG systems. It shows how focused domain knowledge and careful preprocessing can enable powerful multilingual document Q&A systems using modern AI stacks.

