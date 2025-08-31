"""Streamlit UI for Agentic RAG System - Streaming Version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vector_store.vector_store import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .streaming-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Use default URLs
        urls = Config.DEFAULT_URLS
        
        # Process documents
        documents = doc_processor.process_urls(urls)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def stream_response(rag_system, question):
    """Handle streaming response from RAG system"""
    
    start_time = time.time()
    
    # Show retrieval status
    with st.status("🔍 Retrieving documents...", expanded=True) as status:
        st.write("Searching for relevant documents...")
        
        # Retrieve documents first
        docs = rag_system.nodes.retriever.invoke(question)
        st.write(f"Found {len(docs)} relevant documents")
        status.update(label="✅ Documents retrieved", state="complete")
    
    # Show streaming answer
    st.markdown("### 💡 Answer")
    
    # Create the streaming generator function
    def generate_stream():
        """Generator function for streaming"""
        # Prepare context from retrieved docs
        context_parts = []
        for i, doc in enumerate(docs[:5], 1):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            title = meta.get("title") or meta.get("source") or f"doc_{i}"
            context_parts.append(f"[{i}] {title}\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain enough information, you may use your general knowledge but indicate when you're doing so.

Context:
{context}

Question: {question}

Please provide a comprehensive and helpful answer:"""
        
        # Stream from LLM
        for chunk in rag_system.nodes.llm.stream(prompt):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
            elif hasattr(chunk, 'text') and chunk.text:
                yield chunk.text
            elif isinstance(chunk, str):
                yield chunk
    
    # Use Streamlit's write_stream for real-time streaming
    full_response = st.write_stream(generate_stream())
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Show source documents
    if docs:
        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(docs, 1):
                st.text_area(
                    f"Document {i}",
                    doc.page_content[:300] + "...",
                    height=100,
                    disabled=True,
                    key=f"doc_{i}_{int(time.time() * 1000)}_{i}"
                )
    
    st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    
    # Add to history
    st.session_state.history.append({
        'question': question,
        'answer': full_response,
        'time': elapsed_time
    })

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search with streaming
    if submit and question:
        if st.session_state.rag_system:
            # Clear any previous output and start streaming immediately
            stream_response(st.session_state.rag_system, question)
        else:
            st.error("System not initialized. Please refresh the page.")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()