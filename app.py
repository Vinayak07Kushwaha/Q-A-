import streamlit as st
import os
import tempfile
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from huggingface_hub import login
import time


st.set_page_config(page_title="Book Q&A", layout="wide")


st.title(" Book Q&A System")
st.markdown("""
Upload a PDF book and ask questions about its content. 
The app will use RAG (Retrieval-Augmented Generation) to find relevant parts of the book 
and generate accurate answers.
""")


if 'db' not in st.session_state:
    st.session_state.db = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'huggingface_api_key' not in st.session_state:
    st.session_state.huggingface_api_key = ""


with st.sidebar:
    st.header("Configuration")
    
   
    huggingface_api_key = st.text_input("Enter HuggingFace API Key:", 
                                          type="password", 
                                          help="Get a free API key from huggingface.co")
    
    if huggingface_api_key:
        st.session_state.huggingface_api_key = huggingface_api_key
    
   
    model_name = st.selectbox(
        "Select LLM Model:",
        ["google/flan-t5-large", "mistralai/Mistral-7B-Instruct-v0.1", "tiiuae/falcon-7b-instruct"],
        help="Choose a free model available on HuggingFace"
    )
    

    chunk_size = st.slider("Chunk Size:", min_value=100, max_value=2000, value=500)
    chunk_overlap = st.slider("Chunk Overlap:", min_value=0, max_value=500, value=50)


uploaded_file = st.file_uploader("Upload a PDF Book", type="pdf")


def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    
    text = ""
    with open(tmp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    
    os.unlink(tmp_file_path)
    return text


def process_document(text, chunk_size, chunk_overlap):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    
    db = FAISS.from_texts(chunks, embeddings)
    
    return db, chunks


if uploaded_file and not st.session_state.file_processed:
    with st.spinner("Processing your book... This may take a few minutes."):
       
        text = extract_text_from_pdf(uploaded_file)
        
        st.info(f"📄 Document Length: {len(text)} characters | Approximately {len(text.split())} words")
        st.session_state.db, chunks = process_document(text, chunk_size, chunk_overlap)
        st.session_state.file_processed = True
        
        st.success(f"✅ Document processed successfully! Created {len(chunks)} chunks.")


if st.session_state.file_processed:
    st.header("Ask Questions About Your Book")
    
    query = st.text_input("Enter your question:")
    
    if query and st.session_state.huggingface_api_key:
        with st.spinner("Generating answer..."):
            try:
        
                login(token=st.session_state.huggingface_api_key)
                
        
                llm = HuggingFaceHub(
                    repo_id=model_name,
                    huggingfacehub_api_token=st.session_state.huggingface_api_key,  # Explicitly pass API key
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.7,
                        "max_length": 512
                    }
                )
                
                # Create a custom prompt template
                prompt_template = """
                Answer the question based only on the following context:
                
                {context}
                
                Question: {question}
                
                Answer:
                """
                PROMPT = PromptTemplate(
                    template=prompt_template, 
                    input_variables=["context", "question"]
                )
                
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.db.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
        
                response = qa_chain({"query": query})
                
        
                st.header("Answer")
                st.write(response["result"])
                
                
                st.header("Source Passages")
                for i, doc in enumerate(response["source_documents"]):
                    with st.expander(f"Passage {i+1}"):
                        st.write(doc.page_content)
                        
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
    elif query:
        st.warning("Please enter a HuggingFace API key in the sidebar.")


if not st.session_state.file_processed:
    st.info("👆 Upload a PDF book to get started!")
    
    
    with st.expander("How to use this app"):
        st.markdown("""
        ### Step 1: Get a HuggingFace API Key
        1. Create a free account at [HuggingFace](https://huggingface.co/)
        2. Go to your profile and generate an API token
        3. Enter the API token in the sidebar
        
        ### Step 2: Upload a Book
        Upload a PDF book using the file uploader above.
        
        ### Step 3: Ask Questions
        Once the book is processed, you can ask questions about its content.
        """)