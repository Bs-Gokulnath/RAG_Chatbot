import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import shutil 

# Vector storage setup
vector_space_dir = os.path.join(os.getcwd(), "vector_db")
if not os.path.exists(vector_space_dir):
    os.makedirs(vector_space_dir)

# Streamlit page setup
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG Chatbot (Langchain + LLaMa2)")

# Initialize session states
if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "retriever" not in st.session_state:
    st.session_state['retriever'] = None

# File upload
upload_pdf = st.file_uploader("Upload a PDF file", type="pdf", key='upload_pdf')
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if upload_pdf is not None and st.session_state['vectorstore'] is None:
    with st.spinner("Loading PDF..."):
        pdf_path = os.path.join(os.getcwd(), upload_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(upload_pdf.getbuffer())
        st.session_state['pdf_file_path'] = pdf_path

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(vector_space_dir)

        st.session_state['vectorstore'] = vectorstore
        st.session_state['retriever'] = vectorstore.as_retriever(search_kwargs={"k": 5})

        st.success("PDF loaded successfully!")

# Chat section
llm = OllamaLLM(model="llama2")

if st.session_state['retriever'] is not None:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state['retriever'],
        memory=st.session_state['memory'],
        return_source_documents=False
    )

    user_question = st.text_input("Ask a question about the PDF:", key = 'text')

    if user_question:
        with st.spinner("Thinking..."):
            result = qa_chain.run({'question': user_question})
        st.markdown(f"**You:** {user_question}")
        st.markdown(f"**Bot:** {result}")

def del_vectordb(path):
    if os.path.exists(path):
        shutil.rmtress(path)

def del_pdf(path):
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

if st.button("Clear Session"):
    st.session_state.clear()
    st.session_state['retriever'] = None
    st.session_state['vectorstore'] = None
    del_vectordb(vector_space_dir)
    pdf_p = st.session_state.get('pdf_file_path', None)
    del_pdf(pdf_p)
    st.sessoin_state['pdf_file_path'] = None
    for key in ['upload_pdf', 'text']:
        if key in st.session_state:
            del st.session_state[key]
    st.success("Session and pdf cleared successfully!")
    st.rerun()
