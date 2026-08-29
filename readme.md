# Resume RAG — AI-Powered Profile Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that lets users ask natural-language questions about a candidate's resume and LinkedIn activity.

The system retrieves relevant profile information from a **ChromaDB vector store**, combines it with conversation context, and uses **Groq LLMs** to generate grounded answers without relying on information outside the indexed profile.

## ✨ Features

* 🔎 **Semantic + keyword retrieval** for relevant profile information
* 📄 **Resume PDF ingestion** with section-aware chunking
* 💼 **LinkedIn data ingestion** from exported CSV data
* 🧠 **ChromaDB vector database** with cosine similarity
* 🤖 **Groq-powered answer generation**
* 🛡️ **Grounded responses** that refuse unsupported questions instead of hallucinating
* 💬 **Conversation memory** for follow-up questions
* 📚 **Source metadata** attached to retrieved information
* 🎤 **Voice input** using browser Speech Recognition
* 🔊 **Text-to-speech** for AI responses
* 📝 **Markdown-rendered responses**
* ⚡ Query classification for projects, profile overview, entities, and normal questions

## 🏗️ Architecture

```text
                    User
                     │
                     ▼
              Next.js Frontend
                     │
                  REST API
                     │
                     ▼
               FastAPI Backend
                     │
                     ▼
              Query Processing
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Query Analysis         Entity Detection
          │                     │
          └──────────┬──────────┘
                     ▼
              Hybrid Retrieval
                     │
                     ▼
                ChromaDB
                     │
                     ▼
            Relevant Chunks
                     │
                     ▼
             Grounding Guard
                     │
                     ▼
                 Groq LLM
                     │
                     ▼
              Final Answer
```

## 📂 Data Pipeline

The profile is built from two sources:

```text
Resume.pdf
    │
    ▼
PDF Extraction
    │
    ▼
Section-aware Chunking
    │
    ▼
Embeddings
    │
    ▼
ChromaDB
```

```text
LinkedIn CSV
    │
    ▼
LinkedIn Parser
    │
    ▼
Metadata Extraction
    │
    ▼
Embeddings
    │
    ▼
ChromaDB
```

Both sources are stored in the same vector collection, allowing questions to be answered using combined resume and LinkedIn information.

## 🧩 Project Structure

```text
Resume-RAG/
│
├── src/
│   ├── chunk.py
│   ├── conversations.py
│   ├── embeddings.py
│   ├── errors.py
│   ├── extract.py
│   ├── ingest.py
│   ├── linkedin_parser.py
│   ├── prompts.py
│   ├── rag.py
│   ├── retrieve.py
│   ├── section_parser.py
│   └── vector_store.py
│
├── data/
│   ├── Resume.pdf
│   └── linkedin/
│       └── Rich_Media.csv
│
├── frontend/
│   └── ...
│
├── chroma_db/
├── requirements.txt
├── .env.example
└── README.md
```

## 🔍 Retrieval Pipeline

Instead of sending the entire profile to the LLM, the system first determines what information is relevant.

### 1. Query Analysis

The query is analyzed for:

* Query type
* Topics
* Entities
* Relevant keywords

For example:

```text
"What projects has Rohan built?"

→ Query type: project_list
→ Topic: projects
```

### 2. Entity Retrieval

Named entities such as projects, companies, or hackathons can be detected and retrieved directly.

Example:

```text
"What did Rohan work on during Hacknite?"

→ Entity: Hacknite
→ Relevant LinkedIn chunk retrieved
```

### 3. Semantic Retrieval

The query is converted into an embedding and compared against stored vectors using ChromaDB.

### 4. Relevance Filtering

Retrieved chunks are scored using retrieval signals such as:

* Semantic similarity
* Keyword matches
* Entity matches
* Section relevance
* Query intent

Low-confidence results are filtered before reaching the LLM.

### 5. Grounded Generation

Only the retrieved profile context is provided to the LLM.

If the required information is unavailable, the system returns a controlled response rather than generating an unsupported answer.

## 🛡️ Anti-Hallucination Design

One of the main goals of the project is to prevent the model from inventing candidate information.

For example:

```text
User:
What is Rohan's favorite football team?

System:
I don't have that information in the provided profile.
```

The LLM is explicitly instructed to use only retrieved profile context, while the retrieval layer also applies confidence filtering before generation.

This creates two layers of protection:

```text
Retrieval Confidence
        ↓
Context Filtering
        ↓
LLM Grounding Rules
        ↓
Final Answer
```

## 🧠 Conversation Memory

The system supports multi-turn conversations.

For example:

```text
User:
What projects has Rohan built?

AI:
Rohan has worked on JugaadHub, CollabSphere and VideoTube...

User:
Which one used Redis?

AI:
CollabSphere used Redis...
```

The conversation history helps interpret follow-up questions, while retrieved profile context remains the source of truth.

## 💼 LinkedIn Integration

LinkedIn exported data is processed through `linkedin_parser.py`.

The parser:

1. Reads the LinkedIn CSV
2. Removes media-only entries
3. Cleans post text
4. Detects projects
5. Detects hackathons
6. Detects technical technologies
7. Generates metadata
8. Converts posts into RAG-ready documents

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

This allows questions about experiences mentioned on LinkedIn to be answered alongside resume information.

## 🎤 Voice Interaction

The frontend supports browser-based voice interaction.

### Voice Input

```text
Microphone
    ↓
Browser Speech Recognition
    ↓
Transcribed question
    ↓
RAG API
```

### Voice Output

```text
AI Response
    ↓
SpeechSynthesis
    ↓
Browser Audio
```

Voice functionality uses browser-native Web Speech APIs, so no separate speech-processing service is required.

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI
* ChromaDB
* Sentence Transformers
* Groq
* PyPDF
* Python-dotenv

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS
* React Markdown
* Web Speech API

### AI / RAG

* Vector embeddings
* Semantic similarity search
* Hybrid retrieval
* Metadata-aware retrieval
* Groq LLM
* Grounded generation

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/resume-rag.git
cd resume-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
HF_TOKEN=your_huggingface_token
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=resume
```

### 5. Add profile data

Place the resume at:

```text
data/Resume.pdf
```

and LinkedIn exported data at:

```text
data/linkedin/Rich_Media.csv
```

### 6. Build the vector index

```bash
python -m src.ingest
```

Example:

```text
Resume chunks: 9
LinkedIn posts: 10
Total ChromaDB chunks: 19
```

### 7. Test retrieval

```bash
python -m src.retrieve
```

### 8. Start the backend

```bash
uvicorn <your_app_module>:app --reload
```

### 9. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

The application can then be accessed through the local Next.js development server.

## 🔐 Environment Variables

| Variable            | Purpose                        |
| ------------------- | ------------------------------ |
| `GROQ_API_KEY`      | Groq API authentication        |
| `GROQ_MODEL`        | LLM used for answer generation |
| `HF_TOKEN`          | Hugging Face authentication    |
| `CHROMA_PATH`       | Persistent ChromaDB location   |
| `CHROMA_COLLECTION` | ChromaDB collection name       |

**Never commit `.env` or API keys to GitHub.**

## 🧪 Example Questions

```text
What are Rohan's main technical skills?

What projects has Rohan built?

What did Rohan work on during Hacknite?

Did Rohan attend Hacknite?

What is Rohan's academic qualification?

What hackathons has Rohan participated in?

Should we hire Rohan for a backend engineering role?

What is Rohan's favorite football team?
```

The last question demonstrates the system's grounding behavior when information is not present in the indexed profile.

## 🎯 What This Project Demonstrates

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation
* Vector databases
* Embedding-based search
* Hybrid retrieval
* Metadata-aware retrieval
* Query classification
* Context filtering
* LLM grounding
* Conversation memory
* PDF processing
* LinkedIn data processing
* REST API development
* Voice interfaces
* Full-stack AI application development

## 👨‍💻 Author

**Rohan Garg**

B.Tech — Information Technology
University School of Information, Communication and Technology
Guru Gobind Singh Indraprastha University

---

⭐ If you find the project useful, consider giving the repository a star.
