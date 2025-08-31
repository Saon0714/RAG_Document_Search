"""LangGraph nodes for RAG workflow with streaming support"""

from typing import Generator
from src.state.rag_state import RAGState

class RAGNodes:
    """Contains node functions for RAG workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
    
    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents node
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated RAG state with retrieved documents
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from retrieved documents node (non-streaming)
        
        Args:
            state: Current RAG state with retrieved documents
            
        Returns:
            Updated RAG state with generated answer
        """
        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        # Create prompt
        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {state.question}"""
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
    
    def generate_answer_streaming(self, state: RAGState) -> Generator[str, None, RAGState]:
        """
        Generate answer from retrieved documents with streaming
        
        Args:
            state: Current RAG state with retrieved documents
            
        Yields:
            Partial answer chunks as they are generated
            
        Returns:
            Final RAGState with complete answer
        """
        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        # Create prompt
        prompt = f"""Answer the question based on the context.

Context:
{context}

Question: {state.question}"""
        
        # Stream the response
        full_answer = ""
        try:
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_answer += chunk.content
                    yield full_answer
                elif hasattr(chunk, 'text') and chunk.text:
                    full_answer += chunk.text
                    yield full_answer
                elif isinstance(chunk, str):
                    full_answer += chunk
                    yield full_answer
        except Exception as e:
            error_msg = f"Error during streaming: {str(e)}"
            yield error_msg
            full_answer = error_msg
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=full_answer
        )