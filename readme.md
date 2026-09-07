# 🤖 Resume RAG — AI-Powered Profile Assistant

> **Ask questions about Rohan's experience, projects, skills, achievements — or upload your own data and let the RAG system reason over it temporarily.**

Resume RAG is a full-stack **Retrieval-Augmented Generation (RAG)** application that turns a resume and LinkedIn profile into an interactive AI assistant.

Instead of sending the entire profile to an LLM, the system performs **query analysis → embedding-based retrieval → hybrid relevance scoring → context filtering → grounded generation** to provide answers based only on relevant information.

The application also supports **Dynamic RAG**, allowing users to upload PDF, DOCX, Markdown, or CSV files during a session. Uploaded documents are embedded into an isolated temporary ChromaDB collection and can be queried alongside the permanent profile knowledge base.

---

## ✨ Key Features

### 🧠 RAG-Based Profile Assistant

* Ask natural-language questions about the candidate.
* Retrieve relevant information from resume and LinkedIn data.
* Semantic similarity search using MiniLM embeddings.
* ChromaDB vector storage with cosine similarity.
* Groq-powered LLM generation.
* Context-aware multi-turn conversations.

### 🔎 Hybrid Retrieval

The retrieval system does more than simple vector similarity.

Retrieved candidates are scored using multiple signals:

* Semantic similarity
* Keyword/lexical matching
* Query topic relevance
* Section relevance
* Entity matching
* AI/technical relevance
* Session-data relevance

This allows the system to prioritize chunks that are actually useful for the user's query.

### 🎯 Query-Aware Retrieval

Queries are analyzed before retrieval to identify their intent.

The system can distinguish between queries such as:

```text
"What projects has Rohan built?"
        ↓
Project selection query
        ↓
Retrieve project candidates
```

and:

```text
"What are Rohan's technical skills?"
        ↓
Skills query
        ↓
Retrieve skills-related chunks
```

This allows different retrieval strategies to be used for different types of questions.

### 🛡️ Anti-Hallucination / Grounded Generation

The system is designed to avoid inventing information.

If relevant evidence cannot be found, the assistant responds with a controlled answer such as:

```text
I don't have that information in the provided profile.
```

The grounding pipeline works as:

```text
User Query
    ↓
Query Analysis
    ↓
Vector Retrieval
    ↓
Hybrid Scoring
    ↓
Relevance Filtering
    ↓
Grounding Check
    ↓
Groq LLM
    ↓
Grounded Answer
```

### 📚 Resume + LinkedIn Knowledge Base

The permanent knowledge base combines:

* Resume PDF
* LinkedIn exported data
* Project information
* Skills
* Education
* Achievements
* Hackathons
* Professional experiences

Both sources are converted into RAG-ready chunks with metadata.

---

# 🚀 Dynamic RAG

One of the major features of this project is **temporary session-based knowledge**.

Users can click:

```text
+ Add Data
```

and upload additional information while using the assistant.

### Supported file formats

```text
PDF
DOCX
MD
Markdown
CSV
```

The uploaded file is:

```text
Uploaded File
      ↓
Document Parser
      ↓
Text Extraction
      ↓
Chunking
      ↓
MiniLM Embeddings
      ↓
Temporary ChromaDB Collection
```

The temporary data can then be queried alongside the permanent resume data.

---

## 🔐 Isolated Session Storage

Uploaded documents are **not added to the permanent resume collection**.

Every user session receives a unique session ID:

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

This keeps user-uploaded information isolated from the original profile.

### Session lifecycle

```text
Start Session
     ↓
Generate Session ID
     ↓
Create Temporary Collection
     ↓
Upload Files
     ↓
Embed + Store Documents
     ↓
Query Resume + Session Data
     ↓
End Session
     ↓
Delete Temporary Collection
```

When the session ends, the temporary ChromaDB collection is deleted.

The permanent resume collection is never deleted.

---

## 🍪 Session Cookie Design

The browser stores only the session identifier:

```text
rag_session_id
```

The cookie does **not** contain:

* Uploaded document contents
* Embeddings
* Resume information
* ChromaDB data

The session ID allows the backend to identify which temporary collection belongs to the current session.

---

# 🔍 Retrieval Architecture

The core retrieval system follows a multi-stage pipeline.

```text
                         User Query
                             │
                             ▼
                      Query Analysis
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
        Query Classification          Entity Detection
              │                             │
              └──────────────┬──────────────┘
                             ▼
                     Query Embedding
                             │
                             ▼
                  ┌─────────────────────┐
                  │      ChromaDB       │
                  │                     │
                  │ Permanent Resume    │
                  │        +            │
                  │ Temporary Session   │
                  └──────────┬──────────┘
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
                       Final Answer
```

---

# 🧩 Hybrid Retrieval Scoring

A candidate chunk is not selected purely because it has the highest vector similarity.

The retrieval layer considers several signals.

### Semantic similarity

Measures how conceptually similar the query and chunk are.

```text
Query
  ↓
MiniLM
  ↓
384-dimensional embedding
  ↓
Cosine similarity
  ↓
Semantic score
```

### Lexical matching

Important query terms are compared against candidate text.

This helps when exact technical terms matter.

For example:

```text
Query:
"What project used Redis?"

Candidate:
"CollabSphere uses Redis for TTL-based cache eviction."
```

The lexical match reinforces the semantic result.

### Section relevance

The system identifies relevant profile sections such as:

```text
skills
projects
education
experience
achievements
```

and can give additional weight to chunks from the appropriate section.

### Entity matching

Named entities such as:

```text
JugaadHub
CollabSphere
Hacknite
Redis
Razorpay
```

can be used to improve retrieval precision.

### Session relevance

When temporary uploaded documents are present, session chunks receive additional relevance handling so they can participate in retrieval without contaminating the permanent profile.

---

# 📄 Permanent Data Pipeline

The permanent knowledge base is created from the resume and LinkedIn data.

## Resume

```text
Resume.pdf
    ↓
PDF Extraction
    ↓
Section Detection
    ↓
Section-Aware Chunking
    ↓
MiniLM Embeddings
    ↓
ChromaDB
```

## LinkedIn

```text
LinkedIn Export
    ↓
CSV Parsing
    ↓
Post Cleaning
    ↓
Project / Event / Technology Detection
    ↓
Metadata Extraction
    ↓
MiniLM Embeddings
    ↓
ChromaDB
```

The two sources are stored in the same permanent collection.

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

This metadata is later used by the retrieval layer.

---

# 📂 Dynamic Document Pipeline

Temporary documents follow a separate ingestion path:

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

Documents are divided into manageable chunks before embedding.

Current chunking configuration:

```text
Chunk size:    800 characters
Overlap:       120 characters
```

Embeddings are generated in batches to reduce memory usage during document ingestion.

---

# 🗄️ ChromaDB Architecture

The application uses ChromaDB as its vector database.

There are two logical storage layers:

```text
                 ChromaDB
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
      Permanent            Temporary
       Collection            Session
          │                   │
       resume          session_<uuid>
          │                   │
     Resume + LinkedIn    User uploads
```

### Permanent collection

```text
resume
```

Contains the candidate's indexed profile.

### Temporary collection

```text
session_<uuid>
```

Contains only documents uploaded during a particular session.

---

# 🤖 Embeddings

The project uses a MiniLM-based Sentence Transformer model:

```text
sentence-transformers/paraphrase-MiniLM-L3-v2
```

Embedding dimension:

```text
384
```

Embeddings are normalized before being stored and compared.

The same embedding model is used for:

* Resume documents
* LinkedIn documents
* Uploaded session documents
* User queries

This keeps all vectors compatible within ChromaDB.

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

Conversation history helps resolve follow-up questions, while the retrieved profile/session context remains the source of truth.

---

# 🎤 Voice Interaction

The frontend supports browser-native voice interaction.

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

## Voice Output

```text
AI Response
    ↓
SpeechSynthesis
    ↓
Browser Audio
```

No separate speech-processing server is required.

---

# 🎨 Frontend Experience

The application provides a conversational interface for interacting with the RAG system.

The interface includes:

* Chat-based interaction
* Markdown-rendered AI responses
* Voice input
* Text-to-speech
* Dynamic file upload
* Session-aware temporary data
* Upload status handling
* Uploaded-file dropdown

After files are uploaded, the UI keeps track of the current session's input files:

```text
┌──────────────┐
│ 3 files  ▼   │
└──────────────┘
```

Opening the dropdown displays:

```text
Input Data

resume_notes.pdf
project_data.csv
experience.md
```

This provides visibility into what additional data has been added to the current RAG session.

---

# 🔌 Backend API

The backend is implemented using FastAPI.

Main endpoints include:

| Endpoint              | Method | Purpose                               |
| --------------------- | ------ | ------------------------------------- |
| `/api/chat`           | POST   | Send a question to the RAG assistant  |
| `/api/session/start`  | POST   | Start a temporary RAG session         |
| `/api/session/status` | GET    | Check session status                  |
| `/api/session/upload` | POST   | Upload temporary RAG data             |
| `/api/session/end`    | POST   | End session and remove temporary data |

The frontend communicates with the FastAPI backend through REST APIs.

---

# 🏗️ System Architecture

```text
┌─────────────────────────────────────────────┐
│                 Next.js UI                  │
│                                             │
│  Chat │ Voice │ File Upload │ TTS │ UI     │
└──────────────────────┬──────────────────────┘
                       │
                       │ REST API
                       ▼
┌─────────────────────────────────────────────┐
│                 FastAPI                     │
│                                             │
│  Chat API │ Session API │ Upload API        │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│              RAG Service                    │
│                                             │
│ Query Analysis                              │
│ Retrieval                                   │
│ Context Construction                        │
│ Grounding                                   │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐   ┌───────────────┐
│  ChromaDB   │   │ MiniLM        │
│             │   │ Embeddings    │
│ Resume      │   │               │
│ + LinkedIn  │   │ 384 dimensions│
│ + Sessions  │   └───────────────┘
└──────┬──────┘
       │
       ▼
 Relevant Context
       │
       ▼
┌──────────────────┐
│     Groq LLM     │
└────────┬─────────┘
         │
         ▼
   Grounded Answer
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

# 🛠️ Tech Stack

## Frontend

* **Next.js**
* **React**
* **TypeScript**
* **Tailwind CSS**
* **React Markdown**
* **Web Speech API**

## Backend

* **Python**
* **FastAPI**
* **ChromaDB**
* **Sentence Transformers**
* **PyPDF**
* **python-docx**
* **Pandas**
* **Python-dotenv**

## AI / RAG

* **MiniLM embeddings**
* **ChromaDB vector search**
* **Hybrid retrieval**
* **Metadata-aware retrieval**
* **Query classification**
* **Entity detection**
* **Relevance scoring**
* **Context filtering**
* **Groq LLM**
* **Grounded generation**

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

## 2. Create a Python virtual environment

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

Never commit your `.env` file or API keys.

---

# 📥 Build the Permanent Knowledge Base

Place your profile data in:

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

This extracts, chunks, embeds, and stores the permanent profile information in ChromaDB.

---

# ▶️ Run the Backend

Start FastAPI using your application entry point:

```bash
uvicorn <your_app_module>:app --reload
```

The backend will be available at:

```text
http://localhost:8000
```

Swagger documentation:

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

The frontend will be available at:

```text
http://localhost:3000
```

Configure the frontend API URL in `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, this variable should point to the deployed FastAPI backend.

---

# 🧪 Example Questions

### Profile

```text
What are Rohan's main technical skills?

What is Rohan's academic qualification?

What companies has Rohan worked with?

What achievements does Rohan have?
```

### Projects

```text
What projects has Rohan built?

Which project used Redis?

Tell me about CollabSphere.

Which project involved Razorpay?

Compare Rohan's projects.
```

### LinkedIn / Achievements

```text
What did Rohan work on during Hacknite?

Did Rohan participate in hackathons?

What did Rohan contribute to ACM ICPC?
```

### Evaluation

```text
Should we hire Rohan for a backend engineering role?

Would Rohan be suitable for an AI engineering role?
```

### Dynamic RAG

Upload a document and ask:

```text
What does this document contain?

Summarize the uploaded project requirements.

What technologies are mentioned in the uploaded file?

How does this uploaded project compare with Rohan's projects?
```

### Unsupported Information

```text
What is Rohan's favorite football team?
```

Expected behavior:

```text
I don't have that information in the provided profile.
```

---

# 🔐 Security & Data Isolation

Dynamic RAG is designed around temporary session isolation.

### Permanent data

```text
resume
```

remains persistent.

### Session data

```text
session_<uuid>
```

is temporary.

The browser only receives/stores the session identifier needed to associate requests with the temporary collection.

Uploaded documents are not merged into the permanent profile knowledge base.

When the session ends:

```text
Session
   ↓
Temporary Collection
   ↓
Delete
```

This prevents temporary user data from becoming part of the permanent candidate profile.

---

# 🧠 Why RAG Instead of Sending the Entire Resume to the LLM?

A naive implementation could simply send the complete resume and LinkedIn data to the LLM for every question.

This project instead uses retrieval:

```text
Large Profile
     ↓
Vector Database
     ↓
Retrieve Relevant Information
     ↓
Small Context
     ↓
LLM
```

Benefits include:

* Reduced unnecessary context
* Better grounding
* More relevant answers
* Lower token usage
* Better control over hallucination
* Easier source attribution
* Ability to add temporary knowledge dynamically

---

# 🛡️ Grounding Strategy

The application uses multiple layers of protection against hallucination.

```text
             User Query
                  ↓
           Query Analysis
                  ↓
          Candidate Retrieval
                  ↓
           Hybrid Scoring
                  ↓
        Relevance Threshold
                  ↓
        Confidence Evaluation
                  ↓
          Context Filtering
                  ↓
       LLM Grounding Prompt
                  ↓
            Final Answer
```

The LLM is instructed to treat retrieved context as the source of truth rather than inventing candidate information.

---

# 📊 Retrieval Diagnostics

The backend includes detailed retrieval logging to make the RAG pipeline observable.

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

Project retrieval also reports:

```text
Project query embedding completed
dimension=384

Project retrieval collection=resume
count=19

Project retrieval collection=session
count=0

Project retrieval complete
```

This makes it easier to debug:

* Retrieval quality
* Session isolation
* Embedding failures
* Collection state
* Candidate ranking
* Confidence decisions

---

# 💡 Engineering Challenges Solved

This project goes beyond a basic "LLM + vector database" implementation.

### 1. Hybrid retrieval

Pure semantic search can return technically related but contextually incorrect chunks.

The project combines multiple relevance signals to improve precision.

### 2. Query-specific retrieval

Different query types require different retrieval strategies.

Project-list questions, entity questions, overview questions, and normal semantic questions are handled differently.

### 3. Temporary RAG

Users can introduce new knowledge without modifying the permanent knowledge base.

### 4. Session isolation

Each session receives its own ChromaDB collection.

### 5. Controlled generation

Low-confidence retrieval results are filtered before reaching the LLM.

### 6. Memory-conscious embeddings

Embedding models are loaded lazily and document embeddings can be generated in batches, reducing unnecessary memory usage.

### 7. Full-stack integration

The project connects:

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

into one application.

---

# 📈 What This Project Demonstrates

This project demonstrates practical experience with:

* Retrieval-Augmented Generation
* Vector databases
* Embedding models
* Semantic search
* Hybrid retrieval
* Query classification
* Entity detection
* Metadata filtering
* Relevance scoring
* Confidence-based retrieval
* LLM grounding
* Anti-hallucination techniques
* Conversation memory
* Dynamic document ingestion
* Temporary knowledge bases
* Session isolation
* PDF processing
* DOCX processing
* CSV processing
* REST API development
* FastAPI
* Next.js
* TypeScript
* Voice interfaces
* Full-stack AI engineering
* Production deployment

---

# 🔮 Future Improvements

Potential future improvements include:

* [ ] Retrieval evaluation using RAGAS or custom evaluation metrics
* [ ] Automatic source citations in the UI
* [ ] Streaming LLM responses
* [ ] More document formats
* [ ] Persistent user accounts
* [ ] Better session file management
* [ ] File removal from active sessions
* [ ] Retrieval analytics dashboard
* [ ] Automated evaluation dataset
* [ ] Reranking model for improved retrieval precision
* [ ] Background document processing
* [ ] Improved production memory optimization
* [ ] Authentication and authorization
* [ ] Rate limiting
* [ ] Automated cleanup of abandoned sessions

---

# 👨‍💻 Author

## Rohan Garg

**B.Tech — Information Technology**

University School of Information, Communication and Technology
Guru Gobind Singh Indraprastha University

---

## ⭐ Project

Resume RAG is built as a practical demonstration of how modern AI applications can combine **RAG, vector databases, LLMs, document processing, session-based knowledge, and full-stack web development** into a production-oriented system.

If you find the project interesting, consider giving the repository a ⭐.
