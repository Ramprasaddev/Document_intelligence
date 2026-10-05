# 📄 DocuIntelligence — Multi-Model RAG Document Assistant

> A Streamlit-based Retrieval-Augmented Generation (RAG) application that lets users upload documents, automatically index them, ask questions grounded in the uploaded content, and switch between multiple LLM providers without changing the retrieval pipeline.

## 🚀 Overview

**DocuChat** is a document question-answering application built around a modular RAG architecture.

The application accepts:

- PDF
- DOCX
- PPTX
- CSV

After a document is uploaded, DocuChat automatically:

1. Saves the document locally.
2. Detects whether the uploaded content is new using a SHA-256 hash.
3. Loads the document with the appropriate LangChain loader.
4. Splits the content into overlapping chunks.
5. Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
6. Stores the vectors in a persistent Chroma database.
7. Retrieves the most relevant chunks for each question.
8. Sends only the retrieved document context to the selected LLM.
9. Streams the answer back to the user.
10. Displays the retrieved source chunks for transparency.

The key design principle is:

> **One retrieval pipeline, multiple LLM providers.**

This means Azure, Gemini, and OpenRouter can use the same document retrieval layer.

---

## ✨ Features

### 📚 Document Intelligence

- PDF support
- DOCX support
- PPTX support
- CSV support
- Automatic document indexing after upload
- SHA-256 file-content detection to avoid unnecessary re-indexing
- Persistent Chroma vector store
- Source previews for generated answers

### 🧠 RAG Pipeline

- `all-MiniLM-L6-v2` embeddings
- CPU-based embeddings
- Recursive character chunking
- Chunk size: `1000`
- Chunk overlap: `150`
- Similarity retrieval
- Top-K retrieval: `4`
- Strict document-grounded system prompt

### 🤖 Multi-Model Support

The application is designed to support multiple generation providers:

1. **Microsoft Foundry / Azure OpenAI**
   - Deployment configured through `.env`
   - Current project configuration uses `gpt-4.1-mini`

2. **Google Gemini**
   - Google GenAI SDK
   - Model configured through `.env`
   - Recommended to use a currently available Gemini model rather than hard-coding an old model name.

3. **OpenRouter**
   - OpenAI-compatible API
   - Configurable model
   - Supports the `openrouter/free` router for available free models

### 🎨 User Interface

- Claude-inspired clean layout
- Persistent left sidebar
- Automatic document indexing
- Model selector
- Streaming responses
- Source expander
- New chat
- Clear document
- Responsive layout
- Assistant/document-oriented icons

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │       DocuChat       │
                         │      Streamlit UI    │
                         └──────────┬───────────┘
                                    │
                              Upload Document
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Document Loader    │
                         │ PDF / DOCX / PPTX    │
                         │        / CSV         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Text Chunking     │
                         │  Recursive Splitter  │
                         │  Size: 1000          │
                         │  Overlap: 150        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Embeddings       │
                         │ all-MiniLM-L6-v2     │
                         │        CPU           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Chroma         │
                         │   Persistent Vector  │
                         │      Database        │
                         └──────────┬───────────┘
                                    │
                              User Question
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Similarity Retrieval │
                         │       Top-K = 4      │
                         └──────────┬───────────┘
                                    │
                            Retrieved Context
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
       ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
       │ Azure / Foundry│  │ Google Gemini  │  │   OpenRouter   │
       │  GPT-4.1 mini  │  │ Configurable   │  │  Free / Other  │
       └────────────────┘  └────────────────┘  └────────────────┘
                │                   │                   │
                └───────────────────┼───────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   Streaming Answer   │
                         │ + Retrieved Sources  │
                         └──────────────────────┘
```

---

# 🧩 Project Structure

A recommended repository structure is:

```text
DocuChat/
│
├── app.py
├── rag_engine.py
├── requirements.txt
├── .env
├── .gitignore
│
├── uploads/
│   └── uploaded documents
│
├── chroma_db/
│   └── persistent Chroma data
│
└── README.md
```

### `app.py`

Responsible for:

- Streamlit configuration
- UI
- Sidebar
- File upload
- Automatic indexing
- Session state
- Chat history
- Model selection
- Streaming output
- Source rendering

### `rag_engine.py`

Responsible for:

- Document loading
- Chunking
- Embedding generation
- Chroma vector storage
- Similarity retrieval
- Provider clients
- RAG prompt construction
- LLM streaming

### `requirements.txt`

Contains the Python dependencies required by the application.

---

# 🔄 How Automatic Indexing Works

DocuChat intentionally does **not** require an `Index Document` button.

The flow is:

```text
User uploads document
        ↓
SHA-256 hash generated
        ↓
Is this file already indexed?
        │
        ├── YES → Do nothing
        │
        └── NO
             ↓
        Save to uploads/
             ↓
        Load document
             ↓
        Split into chunks
             ↓
        Generate embeddings
             ↓
        Rebuild Chroma collection
             ↓
        Store vectorstore in session
             ↓
        Show "Document indexed successfully"
```

This avoids repeatedly indexing the same file during Streamlit reruns.

---

# 🧠 RAG Implementation

## 1. Document Loading

The application chooses the loader based on the file extension.

| File | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` |
| `.docx` | `Docx2txtLoader` |
| `.pptx` | `UnstructuredPowerPointLoader` |
| `.csv` | `CSVLoader` |

Unsupported extensions are rejected.

---

## 2. Chunking

The project uses:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
```

Why overlap?

Without overlap, useful information near the boundary of two chunks can be separated.

The overlap gives neighboring chunks shared context.

---

## 3. Embeddings

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are generated locally on CPU.

This means the embedding stage does not require a paid embedding API.

---

## 4. Vector Database

The embeddings are stored in:

```text
chroma_db/
```

with collection:

```text
doc_intel
```

Chroma provides persistent vector storage and similarity search.

---

## 5. Retrieval

For each question:

```python
search_type = "similarity"
k = 4
```

The application retrieves the four most relevant document chunks.

The LLM is then given:

```text
System instruction
+
Retrieved document context
+
User question
```

It is **not** given the complete document on every request.

---

# 🛡️ Grounded Answering

DocuChat uses a strict document-grounded system prompt.

The intended behavior is:

```text
Use ONLY supplied document context.
Do not use outside knowledge.
If the answer is not in the document:
"I couldn't find that information in the uploaded document."
```

This is important because a RAG system should not simply answer everything from the model's general knowledge.

The application is designed to reduce hallucination by restricting the answer to retrieved context.

---

# 🤖 LLM Provider Layer

The retrieval layer and generation layer are separated.

```text
                    Chroma
                       │
                       ▼
                Retrieved Chunks
                       │
                       ▼
                Provider Layer
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        Azure        Gemini      OpenRouter
```

This makes it possible to switch the generation provider while keeping:

- document loaders
- chunking
- embeddings
- Chroma
- retrieval
- source display

unchanged.

---

# ☁️ Azure / Microsoft Foundry

Azure is configured using:

```env
AZURE_OPENAI_API_KEY=YOUR_AZURE_KEY
AZURE_OPENAI_ENDPOINT=YOUR_AZURE_ENDPOINT
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
```

The endpoint is normalized to the OpenAI-compatible Azure `/openai/v1` route.

Azure is useful when you want a strong managed enterprise model and already have an Azure subscription or project deployment.

---

# ♊ Google Gemini

Gemini is configured using:

```env
GEMINI_API_KEY=YOUR_GEMINI_KEY
GEMINI_MODEL=YOUR_CURRENTLY_AVAILABLE_GEMINI_MODEL
```

The application uses Google's `google-genai` SDK and streams generated content.

### Important

Do not assume that an older Gemini model name will remain available forever.

If Google reports:

```text
404 NOT_FOUND
model not found
```

check the currently available Gemini models and update:

```env
GEMINI_MODEL=...
```

without changing the RAG pipeline.

---

# 🌐 OpenRouter

OpenRouter uses an OpenAI-compatible client.

Configuration:

```env
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
OPENROUTER_MODEL=openrouter/free
```

The `openrouter/free` router can select an available free model.

Free model availability, limits, and quotas can change, so the application treats the model name as configuration rather than hard-coded RAG logic.

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

Example:

```env
# Microsoft Foundry / Azure
AZURE_OPENAI_API_KEY=YOUR_AZURE_KEY
AZURE_OPENAI_ENDPOINT=YOUR_AZURE_ENDPOINT
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini

# Google Gemini
GEMINI_API_KEY=YOUR_GEMINI_KEY
GEMINI_MODEL=YOUR_CURRENTLY_AVAILABLE_GEMINI_MODEL

# OpenRouter
OPENROUTER_API_KEY=YOUR_OPENROUTER_KEY
OPENROUTER_MODEL=openrouter/free
```

### Never commit `.env`

Add this to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
chroma_db/
uploads/
.streamlit/secrets.toml
```

Never put API keys directly into `app.py` or `rag_engine.py`.

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DocuChat
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Configure `.env`

Create:

```text
.env
```

and add your provider keys.

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Open the URL shown by Streamlit, normally:

```text
http://localhost:8501
```

---

# 🧪 Testing the RAG System

Use a document with clear facts and ask questions whose answers are explicitly present.

For example, for a policy document:

### Test 1 — Scope

```text
Who does this policy apply to?
```

### Test 2 — List retrieval

```text
What are the different types of grievances for internal stakeholders?
```

### Test 3 — Exact fact

```text
What email address should be used to submit a grievance?
```

### Test 4 — Numerical retrieval

```text
How long does the Committee have to acknowledge an internal grievance?
```

### Test 5 — Different section

```text
How long can a complicated grievance take to resolve?
```

### Out-of-context test

A very important RAG test is asking something that the document does not contain:

```text
Who is the CEO of the company?
```

If the document does not contain the answer, the expected behavior is:

```text
I couldn't find that information in the uploaded document.
```

This tests whether the application respects its grounding constraint.

---

# 📊 What to Evaluate

A good RAG evaluation should test more than whether the application produces an answer.

Evaluate:

### 1. Retrieval accuracy

Did the retrieved chunks contain the information needed to answer?

### 2. Answer faithfulness

Did the model answer using the retrieved context?

### 3. Answer relevance

Did it directly answer the question?

### 4. Hallucination resistance

Does it refuse to invent information that is not in the document?

### 5. Source transparency

Does the UI show the document and retrieved source chunks?

### 6. Cross-model consistency

Do Azure, Gemini, and OpenRouter produce answers consistent with the same retrieved context?

---

# ⚠️ Current Limitations

## 1. Scanned PDFs

`PyPDFLoader` works best when the PDF contains extractable text.

A scanned/image-only PDF may require OCR.

## 2. Tables

PDF tables can be difficult to extract perfectly.

Complex layouts may require specialized table extraction or OCR.

## 3. Chroma replacement behavior

The current indexing implementation deletes the existing `doc_intel` collection before creating the new one.

Therefore, the application currently behaves like a **single-document workspace**, not a multi-document knowledge base.

## 4. Local vector storage

Chroma is persisted locally in:

```text
chroma_db/
```

It is not a hosted production vector database.

## 5. Free model limits

Gemini and OpenRouter free availability, quotas, rate limits, and model availability can change.

These are external provider constraints and are not controlled by DocuChat.

## 6. API keys

The application requires provider credentials for cloud LLM providers.

---

# 🔮 Future Improvements

Potential next versions could add:

- Multi-document indexing
- Document deletion without rebuilding everything
- Hybrid keyword + vector retrieval
- Reranking
- Metadata filtering
- Better table extraction
- OCR for scanned PDFs
- Conversation-aware retrieval
- Retrieval evaluation metrics
- RAGAS evaluation
- Token/cost tracking
- Model latency tracking
- Response quality comparison across providers
- Authentication
- User-specific document collections
- Cloud vector database
- Docker deployment
- Azure deployment
- Production logging
- Automated tests
- CI/CD

A particularly valuable improvement would be **multi-document RAG**:

```text
Document A ─┐
Document B ─┼──> Shared Chroma Collection
Document C ─┘
                     ↓
                Retrieval
                     ↓
              Relevant chunks
```

---

# 🧰 Troubleshooting

## `ModuleNotFoundError`

Make sure the virtual environment is activated:

```powershell
.venv\Scripts\activate
```

Then:

```powershell
pip install -r requirements.txt
```

---

## Azure authentication error

Check:

```env
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT
```

Do not expose the key in source code or GitHub.

---

## Gemini 404 / model not found

Check the currently available Gemini models and update:

```env
GEMINI_MODEL=...
```

Then restart Streamlit.

---

## OpenRouter authentication error

Check:

```env
OPENROUTER_API_KEY
OPENROUTER_MODEL
```

Then restart Streamlit.

---

## Document indexing error

Check:

- file extension
- whether the document contains extractable text
- file permissions
- installed dependencies

---

## Chroma problems

Stop Streamlit and, if you intentionally want to rebuild the local vector store, remove:

```text
chroma_db/
```

Then restart the application and upload the document again.

---

# 🔒 Security

Never commit:

```text
.env
API keys
tokens
passwords
private credentials
```

Use environment variables.

If an API key is accidentally exposed publicly:

1. Revoke the key.
2. Create a new key.
3. Update `.env`.
4. Restart the application.

---

# 💡 Why This Project?

Traditional document search forces users to manually search through long documents.

DocuChat changes the interaction model:

```text
Traditional:

Open PDF
   ↓
Ctrl + F
   ↓
Read multiple sections
   ↓
Interpret the information


DocuChat:

Upload document
   ↓
Automatic indexing
   ↓
Ask a question
   ↓
Retrieve relevant context
   ↓
Generate grounded answer
   ↓
Inspect sources
```

The project demonstrates how modern RAG systems combine:

- Document processing
- NLP embeddings
- Vector databases
- Semantic retrieval
- Prompt engineering
- LLM APIs
- Streaming generation
- Multi-provider architecture
- User-facing AI application design

---

# 🧑‍💻 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Language | Python |
| Document processing | LangChain Community |
| PDF | PyPDF |
| DOCX | Docx2txt |
| PPTX | Unstructured / python-pptx |
| CSV | LangChain CSVLoader |
| Text splitting | RecursiveCharacterTextSplitter |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector DB | Chroma |
| Primary LLM | Microsoft Foundry / Azure |
| Secondary LLM | Google Gemini |
| Free model routing | OpenRouter |
| Configuration | python-dotenv |

---

# 📌 Core Design Principle

The most important architectural decision in this project is separating **retrieval** from **generation**.

```text
                  RETRIEVAL
                     │
      ┌──────────────┼──────────────┐
      │              │              │
   Loaders        Embeddings      Chroma
      │              │              │
      └──────────────┼──────────────┘
                     │
              Retrieved Context
                     │
                     ▼
                  GENERATION
                     │
       ┌─────────────┼─────────────┐
       │             │             │
      Azure        Gemini      OpenRouter
       │             │             │
       └─────────────┼─────────────┘
                     │
                     ▼
                 Final Answer
```

Because the two layers are independent, the LLM provider can be changed without redesigning the retrieval system.

---




# 👨‍💻 Author

**Ram Prasad**

AI & Machine Learning Graduate | SAP/Enterprise Technology | Generative AI & RAG

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ and sharing feedback.

Built to explore practical **RAG, multi-model LLM integration, semantic search, and document intelligence**.
