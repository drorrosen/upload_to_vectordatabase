import os
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pathlib import Path
import time
import hashlib
from PyPDF2 import PdfReader
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="CPA Document Upload",
    page_icon="📄",
    layout="wide"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
<style>
    /* RTL Layout */
    body { 
        direction: rtl !important; 
        font-family: 'Heebo', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Text and Typography */
    .stMarkdown, p, h1, h2, h3 {
        text-align: right !important;
        direction: rtl !important;
        color: #E2E8F0 !important;
        line-height: 1.7 !important;
    }
    
    h1, h2, h3 {
        background: linear-gradient(120deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    /* File Uploader */
    .stUploadedFile {
        background: rgba(56, 189, 248, 0.1) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background-color: #38BDF8 !important;
    }
    
    /* Status Messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.2) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Header Container */
    .header-container {
        background: rgba(30, 41, 59, 0.6);
        padding: 2.5rem;
        border-radius: 24px;
        margin: 1.5rem 0 3rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    
    /* Processing Container */
    .processing-container {
        background: rgba(30, 41, 59, 0.6);
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        color: #64748B;
        font-size: 0.9em;
    }
    
    /* Document List Panel */
    .document-list {
        background: rgba(30, 41, 59, 0.6);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 1rem;
    }
    
    .document-item {
        background: rgba(255, 255, 255, 0.15);
        padding: 12px 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-right: 4px solid #38BDF8;
        font-size: 1.1em;
        color: #FFFFFF !important;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        display: block;
        transition: all 0.2s ease;
    }

    .document-item:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateX(-5px);
    }

    /* Sidebar header */
    .sidebar-header {
        color: #38BDF8 !important;
        font-size: 1.3em !important;
        font-weight: bold !important;
        margin-bottom: 15px !important;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# API Key Setup from Secrets
# ---------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = "us-east-1"  # Hardcoded as in app-pinecone.py

# Set Pinecone environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT

def check_secrets():
    """Check if all required secrets are configured"""
    required_secrets = [
        "PINECONE_API_KEY",
        "OPENAI_API_KEY",
    ]
    
    missing_secrets = [secret for secret in required_secrets if secret not in st.secrets]
    
    if missing_secrets:
        st.error("Missing required secrets: " + ", ".join(missing_secrets))
        st.markdown("""
        ### How to configure secrets:
        
        1. Create a `.streamlit/secrets.toml` file in your project directory
        2. Add the following content:
        ```toml
        PINECONE_API_KEY = "your-pinecone-api-key"
        OPENAI_API_KEY = "your-openai-api-key"
        ```
        3. Replace the values with your actual API keys
        
        For Streamlit Cloud:
        1. Go to your app's settings
        2. Find the 'Secrets' section
        3. Add each secret key-value pair
        """)
        st.stop()

def get_safe_id(filename, chunk_idx):
    """Create ASCII-safe ID from filename by hashing Hebrew characters"""
    hash_object = hashlib.md5(filename.encode())
    file_hash = hash_object.hexdigest()[:8]
    return f"doc_{file_hash}_{chunk_idx}"

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text

def upload_single_file(file_path):
    """Upload a single text or PDF file to Pinecone"""
    with st.container():
        st.markdown('<div class="processing-container">', unsafe_allow_html=True)
        
        try:
            # Initialize embeddings model
            embeddings_model = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-3-large"
            )
            
            # Initialize Pinecone with new SDK syntax
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Get the index directly
            index = pc.Index(
                "index",
                host="index-fmrj1el.svc.aped-4627-b74a.pinecone.io"
            )
            
            # Initialize text splitter with smaller chunk size
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=80000,
                chunk_overlap=20000,
                separators=["\n\n=== Document:", "\n\n", "\n", " ", ""]
            )
            
            file_path = Path(file_path)
            
            # Read the content based on file type
            if file_path.suffix.lower() == '.pdf':
                content = extract_text_from_pdf(file_path)
            else:  # Assume text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            chunks = text_splitter.split_text(content)
            st.write(f"Split into {len(chunks)} chunks")
            
            # Create vectors directly for Pinecone
            vectors = []
            for i, chunk in enumerate(chunks):
                embedding = embeddings_model.embed_query(chunk)
                
                # Create vector with minimal metadata
                vector = {
                    'id': f"{file_path.stem}_{i}",
                    'values': embedding,
                    'metadata': {
                        'file': file_path.name,
                        'chunk': i,
                        'total_chunks': len(chunks)
                    }
                }
                vectors.append(vector)
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Upload directly to Pinecone in small batches
            batch_size = 1
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                progress = (i + len(batch)) / len(vectors)
                progress_bar.progress(progress)
                status_text.write(f"מעלה חלקים {i + 1} עד {min(i + batch_size, len(vectors))} מתוך {len(vectors)}")
                
                try:
                    index.upsert(
                        vectors=batch,
                        namespace="Default"
                    )
                    status_text.write("הקבוצה הועלתה בהצלחה")
                except Exception as e:
                    st.error(f"שגיאה בהעלאת קבוצה: {str(e)}")
                    raise e
                
                time.sleep(1)
            
            st.success(f"הקובץ '{file_path.name}' הועלה בהצלחה!")
            st.write(f"סך הכל עובדו {len(vectors)} חלקים")
            
        except Exception as e:
            st.error(f"שגיאה בעיבוד הקובץ: {str(e)}")
            raise
        
        st.markdown('</div>', unsafe_allow_html=True)

def get_available_documents(vector_store):
    """Get list of unique documents in the vector store"""
    try:
        results = vector_store.similarity_search_with_score("", k=1000)
        unique_docs = set()
        for doc, _ in results:
            if 'source' in doc.metadata:
                unique_docs.add(doc.metadata['source'])
        return sorted(list(unique_docs))
    except Exception as e:
        st.error(f"Error fetching documents: {str(e)}")
        return []

def main():
    # Add secrets check at the start of main
    check_secrets()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>📄 העלאת מסמכים למערכת</h1>
        <p style="font-size: 1.2em; color: #94A3B8; margin-top: 1rem;">העלה קבצי PDF או טקסט לעיבוד</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize vector store for document listing
    try:
        embeddings_model = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-large"
        )
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="index",
            embedding=embeddings_model,
            namespace="Default"
        )
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        vector_store = None
    
    # Create single column layout
    uploaded_file = st.file_uploader("בחר קובץ להעלאה", type=['pdf', 'txt'])
    
    if uploaded_file is not None:
        with st.spinner('מעבד את הקובץ...'):
            temp_path = Path(f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            try:
                upload_single_file(temp_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Powered by Pinecone and OpenAI</p>
        <p style="margin-top: 0.5rem;">© 2024 CPA Document Processing</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
