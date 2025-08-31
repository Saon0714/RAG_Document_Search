# 🤖 Agentic RAG System

A powerful Retrieval-Augmented Generation (RAG) system built with **LangGraph**, **Streamlit**, and **ReAct agents** that provides intelligent document search with real-time streaming responses.

## ✨ Features

- **🔍 Intelligent Document Retrieval** - Processes and indexes documents from URLs for semantic search
- **🤖 ReAct Agent Integration** - Uses reasoning and acting agents with tool access (retriever + Wikipedia)
- **🎯 Multi-source Knowledge** - Combines document corpus with Wikipedia for comprehensive answers
- **📊 Interactive UI** - Clean Streamlit interface with search history and source document display
- **🔧 Modular Architecture** - Well-structured codebase with separate components for easy customization

## 🛠️ Tech Stack

- **LangGraph** - Workflow orchestration and agent management
- **LangChain** - Document processing and LLM integration
- **Streamlit** - Interactive web interface
- **Vector Store** - FAISS (Semantic document search)
- **Groq API** - Fast LLM inference (configurable)

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/agentic-rag-system.git
   cd agentic-rag-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   echo "GROQ_API_KEY=your_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📁 Project Structure

```
├── src/
│   ├── config/
│   │   └── config.py              # Configuration settings
│   ├── document_ingestion/
│   │   └── document_processor.py  # Document processing logic
│   ├── vector_store/
│   │   └── vector_store.py        # Vector database management
│   ├── graph_builder/
│   │   └── graph_builder.py       # LangGraph workflow builder
│   ├── nodes/
│   │   ├── nodes.py               # Basic RAG nodes
│   │   └── reactnode.py           # ReAct agent nodes
│   └── state/
│       └── rag_state.py           # State management
├── streamlit_app.py               # Main Streamlit application
├── requirements.txt
└── README.md
```

## 🎯 Use Cases

- **Research Assistant** - Query academic papers and documentation
- **Knowledge Base Search** - Search through company documents and wikis
- **Educational Tool** - Ask questions about learning materials
- **Content Analysis** - Analyze and extract insights from document collections

## 🔧 Customization

- **Add new document sources** - Modify `Config.DEFAULT_URLS`
- **Change LLM provider** - Update `Config.LLM_MODEL`
- **Adjust chunking strategy** - Modify `CHUNK_SIZE` and `CHUNK_OVERLAP`
- **Add new tools** - Extend the ReAct agent with additional tools

## 📝 Example Usage

```python
# Ask questions about your documents
"What are the key concepts in agent architectures?"
"How do diffusion models work for video generation?"
"Compare different approaches to retrieval-augmented generation"
```

The system will:
1. 🔍 Retrieve relevant document chunks
2. 🤖 Generate contextual answers using ReAct reasoning
3. ⚡ Stream responses in real-time
4. 📄 Show source documents for transparency

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built with ❤️ using LangGraph, LangChain, and Streamlit**