import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import tempfile
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI PDF Chat",
    page_icon="📄",
    layout="wide"
)

# -----------------------------------
# CHAT HISTORY
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------
# LOAD CSS
# -----------------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------------
# CACHE EMBEDDING MODEL
# -----------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embeddings()

# -----------------------------------
# LOAD LLM
# -----------------------------------
@st.cache_resource
def load_llm():
    return Ollama(model="phi3")

llm = load_llm()

# -----------------------------------
# TITLE
# -----------------------------------
st.markdown(
    """
    <div class="main-title">
        📄 AI PDF Chat Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload PDFs and chat with your documents using AI locally
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("AI PDF Chat")
st.sidebar.markdown("---")

st.sidebar.write("### Model")
st.sidebar.write("phi3")

st.sidebar.write("### Technologies")
st.sidebar.write("- Streamlit")
st.sidebar.write("- LangChain")
st.sidebar.write("- ChromaDB")
st.sidebar.write("- Ollama")

st.sidebar.markdown("---")
st.sidebar.info(
    "Upload a PDF and ask questions about its content."
)

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.rerun()

# -----------------------------------
# FILE UPLOAD SECTION
# -----------------------------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# PROCESS PDF
# -----------------------------------
if uploaded_file:

    st.success("PDF uploaded successfully!")

    # Save temporary PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        pdf_path = temp_file.name

    # -----------------------------------
    # LOAD PDF
    # -----------------------------------
    with st.spinner("Reading PDF..."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    # -----------------------------------
    # SPLIT TEXT
    # -----------------------------------
    with st.spinner("✂️ Splitting text into chunks..."):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(documents)

    # -----------------------------------
    # CREATE VECTOR STORE
    # -----------------------------------
    with st.spinner("Creating embeddings..."):

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="db"
        )

    # -----------------------------------
    # CREATE QA CHAIN
    # -----------------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    st.markdown("---")

    # -----------------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------------
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # -----------------------------------
    # QUESTION INPUT
    # -----------------------------------
    question = st.chat_input(
    "Ask a question about the PDF..."
    )

    # -----------------------------------
    # GENERATE ANSWER
    # -----------------------------------
    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("🤖 Thinking..."):

                answer = qa_chain.run(question)

            st.markdown(answer)

        st.session_state.messages.append(
             {
                "role": "assistant",
                "content": answer
             }
        )

        with st.spinner("AI is thinking..."):

            answer = qa_chain.run(question)

        st.markdown("## AI Answer")

        st.markdown(
            f"""
            <div class="answer-box">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------
    # CLEANUP
    # -----------------------------------
    os.remove(pdf_path)

# -----------------------------------
# PDF ANALYTICS DASHBOARD
# -----------------------------------

total_pages = len(documents)
total_chunks = len(chunks)

total_words = sum(
    len(doc.page_content.split())
    for doc in documents
)

estimated_read_time = max(1, total_words // 200)

st.markdown("## 📊 Document Analytics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📄 Pages",
        value=total_pages
    )

with col2:
    st.metric(
        label="🧩 Chunks",
        value=total_chunks
    )

with col3:
    st.metric(
        label="📝 Words",
        value=f"{total_words:,}"
    )

with col4:
    st.metric(
        label="⏱ Read Time",
        value=f"{estimated_read_time} min"
    )

st.markdown(
    f"""
    <div class="doc-summary">
        <h3>📚 Uploaded Document</h3>

        <p><strong>File:</strong> {uploaded_file.name}</p>

        <p><strong>Pages:</strong> {total_pages}</p>

        <p><strong>Words:</strong> {total_words:,}</p>

        <p><strong>Estimated Reading Time:</strong> {estimated_read_time} minutes</p>

    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("🔍 Processing Details"):

    st.write(f"📄 Pages Loaded: {total_pages}")
    st.write(f"🧩 Chunks Created: {total_chunks}")
    st.write("🧠 Embeddings Generated")
    st.write("💾 Stored in ChromaDB")
    st.write("🤖 Connected to Ollama")

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")

st.markdown(
    """
    <center>
        <p style='color: gray;'>
            Built with ❤️ using Streamlit + LangChain + Ollama by smndr._.tech
        </p>
    </center>
    """,
    unsafe_allow_html=True
)