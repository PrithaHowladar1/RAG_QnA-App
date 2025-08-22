import asyncio
import nest_asyncio
import streamlit as st
import os


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Fix for Streamlit + async gRPC
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()


GOOGLE_API_KEY = "AIzaSyBkqrTtEd7gtBMrOMEd11g9DwVv3lUa2_Y"

# Initialize Gemini LLM and Embeddings
llm_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Streamlit UI Configuration
st.set_page_config(page_title="Gemini RAG Q&A", page_icon="📂", layout="wide")
st.markdown("<h1 style='text-align: center;'>RAG Q&A with Gemini</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a document and ask questions powered by Gemini and LangChain</p>", unsafe_allow_html=True)

# File Upload
uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "txt", "docx"])

if uploaded_file:
    with st.spinner("Processing file... Please wait"):
        # Save uploaded file locally
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Load document based on file type
        if uploaded_file.name.endswith(".pdf"):
            document_loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.name.endswith(".docx"):
            document_loader = Docx2txtLoader(temp_file_path)
        else:
            document_loader = TextLoader(temp_file_path)

        documents = document_loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        document_chunks = text_splitter.split_documents(documents)

        # Create vector store and retriever
        vector_store = FAISS.from_documents(document_chunks, embedding_model)
        retriever = vector_store.as_retriever()

        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=retriever,
            return_source_documents=True
        )

        st.success("File processed successfully.")

        # User Query Input
        user_question = st.text_input("Ask a question from your file:")
        if st.button("Get Answer"):
            if user_question.strip():
                with st.spinner("Fetching answer..."):
                    response = qa_chain(user_question)

                    # Display Answer
                    st.subheader("Answer")
                    st.write(response["result"])

                    # Display Source Documents
                    st.subheader("Sources / Metadata")
                    for i, source_doc in enumerate(response["source_documents"], start=1):
                        st.markdown(f"Source {i}:")
                        st.write(source_doc.metadata)
                        st.code(source_doc.page_content[:300] + "...")