import os
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import time
import hashlib
from PyPDF2 import PdfReader
import streamlit as st

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
</style>
""", unsafe_allow_html=True)

def check_secrets():
    """Check if all required secrets are configured"""
    required_secrets = [
        "PINECONE_API_KEY",
        "OPENAI_API_KEY",
        "PINECONE_ENVIRONMENT"
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
        PINECONE_ENVIRONMENT = "your-pinecone-environment"  # e.g., "us-east-1-aws"
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
            # Initialize Pinecone with error handling
            pinecone.init(
                api_key=st.secrets["PINECONE_API_KEY"],
                environment=st.secrets["PINECONE_ENVIRONMENT"]
            )
            
            # Test the connection
            try:
                index = pinecone.Index("index")
                # Test if index is accessible
                index.describe_index_stats()
            except Exception as e:
                st.error(f"""
                Error connecting to Pinecone index. Please verify:
                1. Your API key is correct
                2. The environment is correct
                3. The index name 'index' exists in your Pinecone project
                
                Error details: {str(e)}
                """)
                st.stop()
                
            embeddings_model = OpenAIEmbeddings(
                openai_api_key=st.secrets["OPENAI_API_KEY"],  # Use secrets
                model="text-embedding-3-large"
            )
            
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
            
            # Add document separator to help with context
            content = f"\n\n=== Document: {file_path.name}\n\n" + content
            
            chunks = text_splitter.split_text(content)
            st.write(f"Split into {len(chunks)} chunks")
            
            vectors_to_upsert = []
            batch_size = 5
            total_chunks = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    progress = (chunk_idx + 1) / len(chunks)
                    progress_bar.progress(progress)
                    status_text.write(f"מעבד חלק {chunk_idx + 1} מתוך {len(chunks)}")
                    
                    embedding = embeddings_model.embed_query(chunk)
                    safe_id = get_safe_id(file_path.name, chunk_idx)
                    
                    # Ensure metadata matches what the query expects
                    vector = {
                        'id': safe_id,
                        'values': embedding,
                        'metadata': {
                            'source': file_path.name,  # Changed from 'file' to 'source' to match query
                            'page': chunk_idx + 1,     # Add page number for better context
                            'text': chunk,             # Store full chunk text
                            'chunk_idx': chunk_idx,
                            'total_chunks': len(chunks)
                        }
                    }
                    vectors_to_upsert.append(vector)
                    total_chunks += 1
                    
                    if len(vectors_to_upsert) >= batch_size:
                        status_text.write(f"מעלה קבוצה של {len(vectors_to_upsert)} וקטורים...")
                        try:
                            index.upsert(
                                vectors=vectors_to_upsert,
                                namespace="Default"
                            )
                            status_text.write("הקבוצה הועלתה בהצלחה")
                        except Exception as e:
                            if "2MB" in str(e):
                                half_batch = len(vectors_to_upsert) // 2
                                status_text.write(f"מנסה שוב עם קבוצה קטנה יותר של {half_batch}")
                                index.upsert(
                                    vectors=vectors_to_upsert[:half_batch],
                                    namespace="Default"
                                )
                                vectors_to_upsert = vectors_to_upsert[half_batch:]
                                continue
                            else:
                                raise e
                        vectors_to_upsert = []
                        time.sleep(1)  # Reduced sleep time
                    
                except Exception as e:
                    st.error(f"שגיאה בעיבוד חלק: {str(e)}")
                    continue
            
            # Handle remaining vectors
            if vectors_to_upsert:
                try:
                    index.upsert(
                        vectors=vectors_to_upsert,
                        namespace="Default"
                    )
                except Exception as e:
                    if "2MB" in str(e):
                        half_batch = len(vectors_to_upsert) // 2
                        index.upsert(
                            vectors=vectors_to_upsert[:half_batch],
                            namespace="Default"
                        )
                        index.upsert(
                            vectors=vectors_to_upsert[half_batch:],
                            namespace="Default"
                        )
            
            st.success(f"הקובץ '{file_path.name}' הועלה בהצלחה!")
            st.write(f"סך הכל עובדו {total_chunks} חלקים")
            
        except Exception as e:
            st.error(f"שגיאה בעיבוד הקובץ: {str(e)}")
            raise
        
        st.markdown('</div>', unsafe_allow_html=True)

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
