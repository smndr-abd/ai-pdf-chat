# AI PDF Chat Assistant

A modern AI-powered PDF chatbot built with:

* Python
* Streamlit
* LangChain
* ChromaDB
* Ollama
* Sentence Transformers

Upload PDFs and ask questions using a local LLM.

---

# Features

Upload PDF documents

Ask questions about uploaded PDFs

Local AI processing (No OpenAI API)

RAG (Retrieval-Augmented Generation)

Vector database search

Modern Streamlit UI

Local LLM support with Ollama

---

# IMPORTANT

For maximum stability:

Use:

```bash
Python 3.11
```

Do NOT use Python 3.13.

Many AI libraries are still unstable on 3.13.

---

# Recommended Setup

## Install Python 3.11

Download:

[https://www.python.org/downloads/release/python-3110/](https://www.python.org/downloads/release/python-3110/)

Verify:

```bash
python3.11 --version
```

Expected:

```bash
Python 3.11.x
```

---

# Project Structure

```bash
ai_pdf_chat_project/
│
├── app.py
├── README.md
├── requirements.txt
└── venv/
```

---

# Installation Guide

## 1. Clone Repository

```bash
git clone YOUR_REPOSITORY_LINK
```

Move into folder:

```bash
cd ai_pdf_chat_project
```

---

## 2. Create Virtual Environment

```bash
python3.11 -m venv venv
```

---

## 3. Activate Virtual Environment

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 4. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 5. Install Stable Dependencies

Install these EXACT versions.

### Install Streamlit

```bash
python -m pip install streamlit==1.35.0
```

---

### Install LangChain Stable Versions

```bash
python -m pip install langchain==0.1.20
python -m pip install langchain-community==0.0.38
python -m pip install langchain-text-splitters
```

---

### Install AI Packages

```bash
python -m pip install torch==2.2.2
python -m pip install transformers==4.41.2
python -m pip install sentence-transformers==2.7.0
python -m pip install accelerate==0.30.1
```

---

### Install Remaining Packages

```bash
python -m pip install chromadb
python -m pip install pypdf
python -m pip install ollama
```

---

# Install Ollama

Download:

[https://ollama.com/download/mac](https://ollama.com/download/mac)

Install normally.

---

# IMPORTANT — Start Ollama Properly

Open Ollama application manually from:

```bash
Applications → Ollama
```

You should see Ollama icon in the Mac menu bar.

---

# Start Ollama Server

Open terminal:

```bash
ollama serve
```

Expected:

```bash
Listening on 127.0.0.1:11434
```

KEEP THIS TERMINAL OPEN.

---

# Download Llama 3 Model

Open NEW terminal:

```bash
ollama pull llama3
```

Model size:

```bash
~4–5GB
```

---

# Test Ollama

```bash
ollama run llama3
```

Expected:

```bash
>>> Send a message
```

Test:

```bash
hello
```

If AI answers → Ollama works.

---

# IMPORTANT CODE FIX

Open:

```python
app.py
```

Replace this:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

WITH:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

---

# Run Streamlit App

Open NEW terminal:

```bash
source venv/bin/activate
python -m streamlit run app.py
```

Expected:

```bash
Local URL: http://localhost:8501
```

Open URL in browser.

---

# Final Working Architecture

```text
Browser
   ↓
Streamlit UI
   ↓
LangChain
   ↓
ChromaDB
   ↓
Ollama
   ↓
Llama 3
```

---

# Modern Streamlit UI Design

Replace your current `app.py` with this improved version.

```python
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import tempfile
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI PDF Chat",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    .stApp {
        background: linear-gradient(to right, #0f172a, #1e293b);
    }

    h1 {
        color: #38bdf8;
        text-align: center;
        font-size: 3rem;
    }

    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .stButton>button {
        background-color: #38bdf8;
        color: black;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Title
# -----------------------------
st.markdown("<h1>📄 AI PDF Chat Assistant</h1>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Upload PDFs and chat with your documents using AI</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚡ AI PDF Chat")
st.sidebar.write("Built with LangChain + Ollama")

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    st.success("PDF uploaded successfully!")

    # -----------------------------
    # Load PDF
    # -----------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # -----------------------------
    # Split Text
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # -----------------------------
    # Embeddings
    # -----------------------------
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # Vector Store
    # -----------------------------
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="db"
    )

    # -----------------------------
    # Ollama LLM
    # -----------------------------
    llm = Ollama(model="llama3")

    # -----------------------------
    # Retrieval QA
    # -----------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    # -----------------------------
    # User Question
    # -----------------------------
    question = st.text_input("Ask a question about the PDF")

    if question:
        with st.spinner("AI is thinking..."):
            answer = qa_chain.run(question)

        st.markdown("## 🤖 AI Answer")
        st.write(answer)

    os.remove(pdf_path)
```

---

# Example Questions

```text
What is machine learning?
```

```text
Summarize chapter 2
```

```text
What are the key findings?
```

---

# Common Errors + Fixes

## Error

```bash
ModuleNotFoundError
```

Fix:

```bash
python -m pip install PACKAGE_NAME
```

---

## Error

```bash
Connection refused localhost:11434
```

Cause:

Ollama server is not running.

Fix:

```bash
ollama serve
```

Then:

```bash
ollama run llama3
```

---

## Error

```bash
Torch dependency conflicts
```

Cause:

Using Python 3.13.

Fix:

Use Python 3.11.

---

# Future Improvements

* Multiple PDF uploads
* Chat history
* Authentication
* Voice assistant
* Cloud deployment
* Streaming responses
* Better vector search
* OpenAI API support

---

# Resume Description

```text
Built an AI-powered PDF Question Answering system using Python, LangChain, ChromaDB, and local LLMs with Retrieval-Augmented Generation (RAG).
```

---

# Skills Learned

* RAG
* LangChain
* Vector databases
* Embeddings
* Local LLMs
* Streamlit
* AI Engineering
* ChromaDB
* Ollama
