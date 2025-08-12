import streamlit as st
import PyPDF2
from io import BytesIO
import anthropic
import os
from typing import List, Dict
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import uuid

load_dotenv()

st.set_page_config(
    page_title="InkLink",
    page_icon="💬",
    layout="wide"
)

def extract_text_from_pdf(pdf_file) -> str:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

@st.cache_resource
def initialize_vector_db():
    # Use persistent client with a specific path
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="pdf_documents",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def store_document_chunks(collection, embedding_model, filename: str, text: str):
    chunks = chunk_text(text)
    doc_hash = hashlib.md5(f"{filename}_{text[:100]}".encode()).hexdigest()
    
    # Check if document already exists
    try:
        existing_docs = collection.get(
            where={"document_hash": doc_hash}
        )
        
        if existing_docs['ids']:
            return len(chunks)  # Document already processed
    except Exception as e:
        # Collection might be empty or have issues, continue with processing
        pass
    
    embeddings = embedding_model.encode(chunks).tolist()
    
    ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "filename": filename,
            "chunk_index": i,
            "document_hash": doc_hash
        } for i in range(len(chunks))
    ]
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    return len(chunks)

def search_relevant_chunks(collection, embedding_model, query: str, n_results: int = 5) -> List[Dict]:
    try:
        query_embedding = embedding_model.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        relevant_chunks = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                relevant_chunks.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        
        return relevant_chunks
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return []

def initialize_claude():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Please set your ANTHROPIC_API_KEY environment variable")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

def chat_with_pdf_rag(client, collection, embedding_model, user_question: str, chat_history: List[Dict]) -> str:
    # Search for relevant chunks
    relevant_chunks = search_relevant_chunks(collection, embedding_model, user_question, n_results=5)
    
    if not relevant_chunks:
        return "I couldn't find relevant information in the uploaded documents to answer your question."
    
    # Build context from relevant chunks
    context_parts = []
    for chunk in relevant_chunks:
        filename = chunk['metadata']['filename']
        content = chunk['content']
        context_parts.append(f"From {filename}:\n{content}\n")
    
    context = "Based on the following relevant excerpts from your documents, I'll answer your question:\n\n" + "\n---\n".join(context_parts)
    
    messages = []
    # Add recent chat history (last 4 messages to maintain context)
    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    for msg in recent_history:
        if msg["role"] in ["user", "assistant"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    messages.append({
        "role": "user",
        "content": f"{context}\n\nQuestion: {user_question}\n\nPlease provide a comprehensive answer based on the provided excerpts. If the excerpts don't contain enough information to fully answer the question, please mention what additional information might be needed."
    })
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=messages
    )
    
    return response.content[0].text

def main():
    st.title("💬 InkLink")
    st.markdown("*Turn every page into a conversation.*")
    st.divider()
    
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'claude_client' not in st.session_state:
        st.session_state.claude_client = initialize_claude()
    if 'vector_db_initialized' not in st.session_state:
        st.session_state.chroma_client, st.session_state.collection = initialize_vector_db()
        st.session_state.embedding_model = load_embedding_model()
        st.session_state.vector_db_initialized = True
    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0
    
    with st.sidebar:
        st.header("Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or multiple PDF files to chat with"
        )
        
        if uploaded_files:
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs and building vector database..."):
                    total_chunks = 0
                    processed_files = []
                    
                    for uploaded_file in uploaded_files:
                        pdf_text = extract_text_from_pdf(uploaded_file)
                        chunks_added = store_document_chunks(
                            st.session_state.collection,
                            st.session_state.embedding_model,
                            uploaded_file.name,
                            pdf_text
                        )
                        total_chunks += chunks_added
                        processed_files.append(uploaded_file.name)
                    
                    st.session_state.document_count += len(uploaded_files)
                    st.session_state.chat_history = []
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} PDF(s)")
                    st.info(f"📊 Created {total_chunks} searchable chunks from documents")
                    
                    with st.expander("📁 Processed Files"):
                        for filename in processed_files:
                            st.write(f"• {filename}")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
            
        if st.button("🗑️ Clear All Documents"):
            # Clear the vector database
            try:
                st.session_state.chroma_client.delete_collection("pdf_documents")
            except:
                pass  # Collection might not exist
            st.session_state.chroma_client, st.session_state.collection = initialize_vector_db()
            st.session_state.document_count = 0
            st.session_state.chat_history = []
            st.success("All documents cleared from database")
            st.rerun()
            
        # Show database stats
        if st.session_state.document_count > 0:
            st.divider()
            st.markdown("### 📊 Database Stats")
            try:
                collection_count = st.session_state.collection.count()
                st.metric("Documents Processed", st.session_state.document_count)
                st.metric("Total Chunks", collection_count)
            except:
                st.metric("Documents Processed", st.session_state.document_count)
                st.metric("Total Chunks", "Loading...")
    
    if st.session_state.document_count > 0:
        st.success(f"🚀 {st.session_state.document_count} PDF(s) loaded in vector database! Advanced RAG-powered chat is ready.")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if prompt := st.chat_input("Ask a question about your PDF(s)"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents and generating response..."):
                    response = chat_with_pdf_rag(
                        st.session_state.claude_client,
                        st.session_state.collection,
                        st.session_state.embedding_model,
                        prompt,
                        st.session_state.chat_history[:-1]
                    )
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Footer after chat input
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 12px; margin-top: 1rem; padding: 10px;'>
                <p>Made with ❤️ by <strong>Geetanshi Goel</strong></p>
                <div style='margin-top: 8px;'>
                    <a href="https://github.com/geetanshi0205" target="_blank" style='margin: 0 8px; text-decoration: none;'>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="vertical-align: middle;">
                            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.30.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                        </svg>
                    </a>
                    <a href="https://www.linkedin.com/in/geetanshi-goel-49ba5832b/" target="_blank" style='margin: 0 8px; text-decoration: none;'>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" style="vertical-align: middle;">
                            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                        </svg>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.info("📤 Please upload and process PDF files to start chatting with RAG-powered AI.")
        
        # Show RAG benefits
        st.markdown(
            """
            ### 🎯 Enhanced with RAG Technology
            
            **What makes InkLink special:**
            - 🔍 **Smart Search**: Finds the most relevant parts of your documents
            - 🧠 **Context-Aware**: Only uses pertinent information for each question
            - ⚡ **Efficient**: Processes large documents by chunking and indexing
            - 🎯 **Accurate**: Reduces hallucinations by grounding responses in your content
            """
        )

if __name__ == "__main__":
    main()