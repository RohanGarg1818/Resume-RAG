# 🤖 Resume RAG — AI-Powered Profile Assistant

> **Ask questions about Rohan's experience, projects, skills, achievements, and more — or upload your own documents and temporarily extend the assistant's knowledge base.**

**Resume RAG** is a full-stack **Retrieval-Augmented Generation (RAG)** application that transforms a resume and LinkedIn profile into an interactive AI-powered profile assistant.

Rather than sending an entire profile to an LLM for every question, the system performs **query analysis, vector retrieval, hybrid relevance scoring, context filtering, grounding, and controlled generation** to produce answers based on relevant source information.

The application also implements **Dynamic RAG**, allowing users to upload PDF, DOCX, Markdown, or CSV files during a session. Uploaded documents are processed, embedded, and stored in an isolated temporary ChromaDB collection without modifying the permanent profile knowledge base.

---

# ✨ Features

## 🧠 AI-Powered Profile Assistant

* Natural-language questions about the candidate
* Resume and LinkedIn knowledge retrieval
* Semantic search using MiniLM embeddings
* ChromaDB vector storage
* Groq-powered LLM generation
* Multi-turn conversational context
* Grounded responses based on retrieved evidence
* Controlled responses when information is unavailable

---

## 🔎 Hybrid Retrieval Engine

The system goes beyond basic vector similarity.

Each retrieved candidate can be evaluated using multiple relevance signals:

* **Semantic similarity**
* **Lexical / keyword matching**
* **Query topic relevance**
* **Section relevance**
* **Entity matching**
* **AI / technical relevance**
* **Session-data relevance**

This allows the system to prioritize information that is not only semantically similar, but also directly useful for the user's specific question.

### Example

```text
User Query
    │
    ▼
"What project used Redis?"
    │
    ▼
Semantic Retrieval
    │
    ▼
Keyword Match → "Redis"
    │
    ▼
Entity / Project Match
    │
    ▼
Hybrid Score
    │
    ▼
Relevant Project Chunk
```

---

# 🎯 Query-Aware Retrieval

Queries are analyzed before retrieval so that different types of questions can use different retrieval strategies.

The system identifies query intent such as:

* Project queries
* Profile / overview queries
* Entity-specific queries
* Evaluation queries
* General semantic queries
* Contact-related queries

For example:

```text
"What projects has Rohan built?"
        │
        ▼
Project Query
        │
        ▼
Project-focused retrieval
```

while:

```text
"What are Rohan's technical skills?"
        │
        ▼
Skills / Profile Query
        │
        ▼
Relevant profile sections
```

This improves retrieval precision compared with treating every question as a generic semantic-search request.

---

# 🛡️ Grounded Generation & Hallucination Control

The system is designed to prevent the LLM from inventing profile information.

The retrieval pipeline applies relevance and confidence checks before context is passed to the LLM.

```text
User Query
    ↓
Query Analysis
    ↓
Candidate Retrieval
    ↓
Hybrid Scoring
    ↓
Relevance Filtering
    ↓
Confidence Check
    ↓
Context Construction
    ↓
Groq LLM
    ↓
Grounded Response
```

When sufficient evidence cannot be retrieved, the system can return a controlled response such as:

```text
I don't have that information in the provided profile.
```

The LLM is instructed to treat retrieved information as the source of truth.

---

# 📚 Permanent Knowledge Base

The assistant's permanent knowledge base combines information from:

* Resume PDF
* LinkedIn exported data
* Projects
* Technical skills
* Education
* Achievements
* Hackathons
* Professional experience

Both resume and LinkedIn data are processed into structured, searchable chunks with metadata.

```text
Resume.pdf ──────────────┐
                         │
                         ▼
                    Processing
                         │
                         ▼
                    Chunking
                         │
                         ▼
                 MiniLM Embeddings
                         │
                         ▼
                      ChromaDB
                         ▲
                         │
LinkedIn CSV ────────────┘
```

---

# 🚀 Dynamic RAG

A major feature of the application is **session-based Dynamic RAG**.

Users can click:

```text
+ Add Data
```

and upload additional documents while interacting with the assistant.

### Supported formats

```text
PDF
DOCX
MD
Markdown
CSV
```

Uploaded documents follow a separate ingestion pipeline:

```text
Uploaded File
      ↓
Document Loader
      ↓
Text Extraction
      ↓
Chunking
      ↓
MiniLM Embeddings
      ↓
Temporary ChromaDB Collection
      ↓
Session Retrieval
```

The uploaded information can then be queried alongside the permanent resume knowledge base.

### Example

```text
Upload: project_requirements.pdf

        ↓

"What technologies are mentioned in this document?"

        ↓

"What does this project require?"

        ↓

"How does this project compare with Rohan's projects?"
```

---

# 🔐 Session Isolation

Dynamic RAG does **not** modify the permanent profile database.

Every user session receives a unique session identifier.

```text
UUID
  ↓
session_<uuid>
  ↓
Temporary ChromaDB Collection
```

For example:

```text
session_a098cffd421540439655e8e98b31dbcee
```

The permanent collection remains:

```text
resume
```

### Storage architecture

```text
                    ChromaDB
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        Permanent            Temporary
        Collection             Session
             │                   │
          resume          session_<uuid>
             │                   │
     Resume + LinkedIn       User uploads
```

This ensures that temporary user data does not contaminate the permanent candidate profile.

---

# 🔄 Session Lifecycle

```text
Start Session
     ↓
Generate Session ID
     ↓
Create / Access Temporary Collection
     ↓
Upload Documents
     ↓
Extract & Chunk Data
     ↓
Generate Embeddings
     ↓
Store in Session Collection
     ↓
Query Permanent + Session Data
     ↓
End Session
     ↓
Delete Temporary Collection
```

When the session ends, its temporary ChromaDB collection is deleted.

The permanent `resume` collection remains intact.

---

# 🍪 Session Cookie Design

The browser stores only the session identifier:

```text
rag_session_id
```

The cookie does not contain:

* Uploaded document contents
* Embeddings
* Resume information
* ChromaDB data

The session identifier allows the backend to associate requests with the correct temporary ChromaDB collection.

---

# 🔍 Retrieval Architecture

The core retrieval pipeline is designed as a multi-stage process.

```text
                         User Query
                             │
                             ▼
                      Query Analysis
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
        Query Classification        Entity Detection
                │                         │
                └────────────┬────────────┘
                             ▼
                     Query Embedding
                             │
                             ▼
                    ┌─────────────────┐
                    │    ChromaDB     │
                    │                 │
                    │ Permanent Data  │
                    │       +         │
                    │ Session Data   │
                    └────────┬────────┘
                             ▼
                     Candidate Chunks
                             │
                             ▼
                      Hybrid Scoring
                             │
                             ▼
                   Relevance Filtering
                             │
                             ▼
                    Confidence Check
                             │
                             ▼
                    Context Builder
                             │
                             ▼
                         Groq LLM
                             │
                             ▼
                      Final Response
```

---

# 🧩 Hybrid Relevance Scoring

Candidate chunks are not selected solely based on vector similarity.

The retrieval system combines several signals.

## Semantic Similarity

The user query is converted into an embedding and compared against document embeddings using cosine similarity.

```text
Query
  ↓
MiniLM
  ↓
384-dimensional vector
  ↓
Cosine Similarity
  ↓
Semantic Score
```

---

## Lexical / Keyword Matching

Exact query terms are also considered.

For example:

```text
Query:
"What project used Redis?"

Candidate:
"CollabSphere uses Redis for TTL-based cache eviction."
```

The exact presence of `Redis` strengthens the relevance of the candidate.

This is particularly useful for:

* Technologies
* Project names
* Company names
* Tools
* Libraries
* Frameworks

---

## Section Relevance

The retrieval layer can use profile sections to improve ranking.

Examples include:

```text
Skills
Projects
Education
Experience
Achievements
```

A project-related question can therefore prioritize chunks from project sections.

---

## Entity Matching

Named entities are used as additional retrieval signals.

Examples:

```text
JugaadHub
CollabSphere
Hacknite
Redis
Razorpay
ACM ICPC
```

Entity matching helps distinguish between conceptually similar chunks that refer to different projects, technologies, or events.

---

## Session Relevance

Temporary session documents participate in retrieval without being merged into the permanent collection.

This enables queries that combine:

```text
Permanent Profile Knowledge
        +
Temporary Uploaded Knowledge
```

while maintaining storage isolation.

---

# 📄 Document Processing

## Resume Processing

```text
Resume.pdf
    ↓
PDF Text Extraction
    ↓
Section Detection
    ↓
Section-Aware Chunking
    ↓
MiniLM Embeddings
    ↓
ChromaDB
```

---

## LinkedIn Processing

```text
LinkedIn Export
    ↓
CSV Parsing
    ↓
Data Cleaning
    ↓
Content / Project / Event Detection
    ↓
Metadata Extraction
    ↓
MiniLM Embeddings
    ↓
ChromaDB
```

Example metadata:

```json
{
  "source": "linkedin",
  "source_type": "linkedin",
  "section": "achievements",
  "event": "Hacknite",
  "project": "JugaadHub"
}
```

Metadata can subsequently be used by the retrieval layer.

---

# 📂 Dynamic Document Processing

Temporary documents follow the same embedding space as the permanent knowledge base.

```text
PDF / DOCX / MD / CSV
          ↓
     Document Loader
          ↓
     Text Extraction
          ↓
        Chunking
          ↓
   MiniLM Embeddings
          ↓
Temporary Chroma Collection
```

### Current chunking configuration

```text
Chunk Size : 800 characters
Overlap    : 120 characters
```

Embedding generation is performed in batches to reduce memory consumption during ingestion.

---

# 🗄️ ChromaDB Architecture

ChromaDB is used as the application's vector database.

The system maintains two logical storage layers:

```text
                       ChromaDB
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
         Permanent                 Temporary
         Collection                Collections
              │                       │
           resume              session_<uuid>
              │                       │
      Resume + LinkedIn         Uploaded files
```

### Permanent collection

```text
resume
```

Contains the indexed candidate profile.

### Temporary collection

```text
session_<uuid>
```

Contains data uploaded during the current session.

---

# 🤖 Embedding Model

The project uses:

```text
sentence-transformers/paraphrase-MiniLM-L3-v2
```

Embedding dimension:

```text
384
```

The same embedding model is used for:

* Resume documents
* LinkedIn documents
* Dynamic session documents
* User queries

This keeps all vectors compatible within the same retrieval system.

---

# 💬 Conversation Memory

The assistant supports multi-turn conversations.

Example:

```text
User:
What projects has Rohan built?

AI:
Rohan has built JugaadHub, CollabSphere, VideoTube...

User:
Which one used Redis?

AI:
CollabSphere used Redis for cache eviction...
```

Conversation history helps interpret follow-up questions, while retrieved documents remain the source of truth for factual answers.

---

# 🎤 Voice Interaction

The frontend supports browser-native voice interaction using the **Web Speech API**.

## Voice Input

```text
Microphone
    ↓
Browser Speech Recognition
    ↓
Transcribed Query
    ↓
RAG API
    ↓
AI Response
```

Users can speak questions instead of typing them.

---

## Text-to-Speech

Assistant responses can also be played using browser-native speech synthesis.

```text
AI Response
    ↓
SpeechSynthesis
    ↓
Browser Audio
```

No dedicated speech-processing backend is required for these browser-native voice features.

---

# 🎨 Frontend Experience

The application provides a conversational interface designed around interacting with the RAG system.

### Implemented UI features

* Chat interface
* Markdown-rendered responses
* Code and structured Markdown rendering
* Voice input
* Text-to-speech
* Dynamic file upload
* Session-aware data
* Uploaded-file tracking
* Uploaded-file dropdown
* Loading states
* Error handling
* Suggested questions
* Multi-turn conversation interface

### Dynamic data interface

Users can add temporary knowledge using:

```text
+ Add Data
```

The interface tracks uploaded files during the current session.

Example:

```text
┌─────────────────┐
│ 3 files      ▼  │
└─────────────────┘
```

Opening the dropdown exposes the currently uploaded input data:

```text
Input Data

resume_notes.pdf
project_data.csv
experience.md
```

---

# 🔌 Backend API

The backend is implemented with **FastAPI** and exposes REST endpoints for chat and session management.

| Endpoint              | Method | Purpose                                   |
| --------------------- | ------ | ----------------------------------------- |
| `/api/chat`           | POST   | Send a question to the RAG assistant      |
| `/api/session/start`  | POST   | Start a temporary RAG session             |
| `/api/session/status` | GET    | Check the current session                 |
| `/api/session/upload` | POST   | Upload temporary RAG documents            |
| `/api/session/end`    | POST   | End the session and delete temporary data |

The frontend communicates with the FastAPI service through REST APIs.

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────┐
│                  Next.js                     │
│                                              │
│ Chat │ Voice │ Upload │ TTS │ Markdown UI    │
└──────────────────────┬───────────────────────┘
                       │
                       │ REST API
                       ▼
┌──────────────────────────────────────────────┐
│                  FastAPI                     │
│                                              │
│ Chat API │ Session API │ Upload API          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│                  RAG Layer                   │
│                                              │
│ Query Analysis                               │
│ Classification                               │
│ Entity Detection                             │
│ Retrieval                                    │
│ Hybrid Scoring                               │
│ Relevance Filtering                          │
│ Grounding                                    │
│ Context Construction                         │
└──────────────┬───────────────────┬───────────┘
               │                   │
               ▼                   ▼
       ┌──────────────┐    ┌─────────────────┐
       │   ChromaDB   │    │     MiniLM      │
       │              │    │   Embeddings    │
       │ Resume       │    │                 │
       │ LinkedIn     │    │ 384 dimensions  │
       │ Sessions     │    └─────────────────┘
       └──────┬───────┘
              │
              ▼
       Relevant Context
              │
              ▼
       ┌──────────────┐
       │    Groq LLM  │
       └──────┬───────┘
              │
              ▼
       Grounded Response
```

---

# 📁 Project Structure

```text
Resume-RAG/
│
├── src/
│   ├── chunk.py
│   ├── conversations.py
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── errors.py
│   ├── extract.py
│   ├── ingest.py
│   ├── linkedin_parser.py
│   ├── prompts.py
│   ├── rag.py
│   ├── retrieve.py
│   ├── section_parser.py
│   ├── session_ingestion.py
│   └── vector_store.py
│
├── backend/
│   └── ...
│
├── frontend/
│   ├── app/
│   ├── components/
│   │   ├── ChatWindow.tsx
│   │   ├── ChatMessage.tsx
│   │   └── DynamicDataUpload.tsx
│   ├── lib/
│   │   └── session.ts
│   └── ...
│
├── data/
│   ├── Resume.pdf
│   └── linkedin/
│       └── Rich_Media.csv
│
├── chroma_db/
│
├── requirements.txt
├── .env.example
└── README.md
```

---

# 🛠️ Technology Stack

## Frontend

* **Next.js**
* **React**
* **TypeScript**
* **Tailwind CSS**
* **React Markdown**
* **remark-gfm**
* **Web Speech API**

## Backend

* **Python**
* **FastAPI**
* **ChromaDB**
* **Sentence Transformers**
* **PyPDF**
* **python-docx**
* **Pandas**
* **python-dotenv**

## AI / RAG

* **MiniLM embeddings**
* **ChromaDB vector search**
* **Hybrid retrieval**
* **Metadata-aware retrieval**
* **Query classification**
* **Entity detection**
* **Lexical matching**
* **Section-aware retrieval**
* **Relevance scoring**
* **Confidence filtering**
* **Grounded generation**
* **Groq LLM**

## Deployment

* **Vercel** — Frontend
* **Render** — Backend

---

# ⚙️ Local Setup

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/resume-rag.git
cd resume-rag
```

---

## 2. Create a virtual environment

```bash
python -m venv venv
```

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
HF_TOKEN=your_huggingface_token
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=resume
```

> Never commit `.env` files or API keys to the repository.

---

# 📥 Build the Permanent Knowledge Base

Place the profile data inside:

```text
data/

├── Resume.pdf
└── linkedin/
    └── Rich_Media.csv
```

Then run:

```bash
python -m src.ingest
```

The ingestion pipeline will:

```text
Extract
  ↓
Clean
  ↓
Parse
  ↓
Chunk
  ↓
Embed
  ↓
Store
```

The resulting profile knowledge is stored in the permanent ChromaDB collection.

---

# ▶️ Run the Backend

Start the FastAPI application using the project's configured application entry point:

```bash
uvicorn <your_app_module>:app --reload
```

Backend:

```text
http://localhost:8000
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

---

# ▶️ Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:3000
```

Configure `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, set this variable to the deployed FastAPI backend URL.

---

# 🧪 Example Queries

## Profile

```text
What are Rohan's main technical skills?

What is Rohan's academic qualification?

What companies has Rohan worked with?

What achievements does Rohan have?
```

## Projects

```text
What projects has Rohan built?

Which project used Redis?

Tell me about CollabSphere.

Which project involved Razorpay?

Compare Rohan's projects.
```

## LinkedIn & Achievements

```text
What did Rohan work on during Hacknite?

Did Rohan participate in hackathons?

What did Rohan contribute to ACM ICPC?
```

## Evaluation

```text
Should we hire Rohan for a backend engineering role?

Would Rohan be suitable for an AI engineering role?
```

## Contact

```text
How can I contact Rohan?

What is Rohan's email?

What is Rohan's phone number?
```

## Dynamic RAG

After uploading a document:

```text
What does this document contain?

Summarize the uploaded project requirements.

What technologies are mentioned in the uploaded file?

How does this project compare with Rohan's projects?
```

## Unsupported Information

```text
What is Rohan's favorite football team?
```

Expected behavior:

```text
I don't have that information in the provided profile.
```

---

# 🔐 Security & Data Isolation

The Dynamic RAG architecture separates permanent and temporary knowledge.

### Permanent data

```text
resume
```

Persistent candidate information.

### Temporary data

```text
session_<uuid>
```

Session-specific uploaded information.

The browser stores only the session identifier required to associate requests with the temporary collection.

Uploaded documents are **not merged into the permanent resume collection**.

When a session ends:

```text
Session
   ↓
Temporary Collection
   ↓
Delete
```

This prevents temporary information from becoming part of the permanent candidate knowledge base.

---

# 🧠 Why RAG?

A basic implementation could send the entire resume and LinkedIn data to the LLM with every request.

Resume RAG instead retrieves only the information relevant to the current question.

```text
Complete Profile
       ↓
   Vector Store
       ↓
Relevant Retrieval
       ↓
Small Context
       ↓
      LLM
       ↓
Grounded Answer
```

This provides several advantages:

* More relevant context
* Better grounding
* Reduced unnecessary token usage
* Better control over hallucinations
* Query-specific retrieval
* Metadata-aware search
* Support for temporary knowledge
* Separation between permanent and temporary data

---

# 📊 Retrieval Observability

The backend includes retrieval diagnostics to make the RAG pipeline easier to inspect and debug.

Example:

```text
Candidate #9

source=session
score=0.611
semantic=0.164
keywords=['rag', 'the']
session=<session-id>

Final retrieval:
kept=1
confidence=high
best_score=1.004
```

Project retrieval can also expose information such as:

```text
Project query embedding completed
dimension=384

Project retrieval collection=resume
count=19

Project retrieval collection=session
count=0

Project retrieval complete
```

These diagnostics help investigate:

* Retrieval quality
* Candidate ranking
* Embedding dimensions
* Session isolation
* Collection state
* Confidence decisions
* Retrieval failures

---

# 💡 Engineering Challenges Addressed

This project was designed to solve problems that appear in practical RAG systems rather than implementing only a basic vector-search chatbot.

### 1. Hybrid Retrieval

Semantic similarity alone can retrieve conceptually related but contextually incorrect information.

The system combines semantic, lexical, entity, section, and topic signals.

### 2. Query-Specific Retrieval

Different questions require different retrieval strategies.

Project, entity, profile, evaluation, and general queries can be handled differently.

### 3. Dynamic Knowledge Injection

Users can introduce new documents without modifying the permanent knowledge base.

### 4. Session Isolation

Each session receives its own temporary ChromaDB collection.

### 5. Grounded Generation

Low-confidence or insufficient retrieval results can be filtered before reaching the LLM.

### 6. Exact Information Retrieval

Queries involving exact profile information, such as contact details, can use targeted retrieval logic rather than relying exclusively on semantic similarity.

### 7. Memory-Conscious Embedding

Embedding models are loaded lazily and documents can be embedded in batches to reduce memory usage.

### 8. Full-Stack AI Integration

The application combines:

```text
Next.js
    +
FastAPI
    +
ChromaDB
    +
MiniLM
    +
Groq
```

into a complete web-based AI application.

---

# 📈 What This Project Demonstrates

### AI / RAG

* Retrieval-Augmented Generation
* Vector databases
* Embedding models
* Semantic search
* Hybrid retrieval
* Query classification
* Entity detection
* Metadata-aware retrieval
* Lexical matching
* Section-aware retrieval
* Relevance scoring
* Confidence filtering
* Context construction
* LLM grounding
* Hallucination mitigation

### Document Intelligence

* PDF processing
* DOCX processing
* Markdown processing
* CSV processing
* Document chunking
* Metadata extraction
* Dynamic document ingestion

### Backend Engineering

* FastAPI
* REST API design
* Session management
* Temporary data isolation
* Error handling
* Retrieval diagnostics
* Environment configuration

### Frontend Engineering

* Next.js
* React
* TypeScript
* Tailwind CSS
* Markdown rendering
* Browser voice input
* Browser text-to-speech
* Dynamic file upload
* Session-aware UI

### Deployment

* Vercel
* Render
* Frontend/backend separation
* Environment-based configuration
* CORS configuration

---

# 🔮 Future Improvements

The following features are planned rather than currently implemented:

* [ ] RAG evaluation using RAGAS or custom metrics
* [ ] Automated evaluation dataset
* [ ] Retrieval analytics dashboard
* [ ] Streaming LLM responses
* [ ] Automatic source citations in the UI
* [ ] Reranking model
* [ ] Background document processing
* [ ] File removal from active sessions
* [ ] Persistent user accounts
* [ ] Authentication and authorization
* [ ] Rate limiting
* [ ] Automated cleanup of abandoned sessions
* [ ] Additional document formats
* [ ] Further production memory optimization

---

# 👨‍💻 Author

## Rohan Garg

**B.Tech — Information Technology**

University School of Information, Communication and Technology
Guru Gobind Singh Indraprastha University

---

# ⭐ Project

**Resume RAG** demonstrates how modern AI engineering techniques can be combined into a complete production-oriented application.

It brings together:

```text
RAG
+
Hybrid Retrieval
+
Vector Search
+
LLM Grounding
+
Dynamic Documents
+
Session Isolation
+
Conversation Memory
+
Voice Interaction
+
Full-Stack Development
```

The result is an interactive AI assistant capable of answering questions about a structured personal knowledge base while also allowing users to temporarily introduce and query their own information.

If you find the project useful or interesting, consider giving the repository a ⭐.
