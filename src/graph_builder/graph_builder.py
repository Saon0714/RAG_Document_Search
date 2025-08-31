"""Graph builder for LangGraph workflow with streaming support"""

from typing import Generator
from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.reactnode import RAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Build the RAG workflow graph
        
        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(RAGState)
        
        # Add nodes
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # Set entry point
        builder.set_entry_point("retriever")
        
        # Add edges
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        
        # Compile graph
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow (non-streaming)
        
        Args:
            question: User question
            
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)
    
    def run_streaming(self, question: str) -> Generator[dict, None, None]:
        """
        Run the RAG workflow with streaming support
        
        Args:
            question: User question
            
        Yields:
            Dictionary containing streaming updates:
            - step: 'retrieving' | 'generating' | 'complete'
            - content: retrieved docs or answer chunk
            - state: current RAGState
        """
        # Step 1: Retrieve documents
        yield {
            'step': 'retrieving', 
            'content': 'Retrieving relevant documents...', 
            'state': None
        }
        
        # Retrieve documents
        docs = self.nodes.retriever.invoke(question)
        state_after_retrieval = RAGState(
            question=question,
            retrieved_docs=docs
        )
        
        yield {
            'step': 'retrieving', 
            'content': f'Found {len(docs)} relevant documents', 
            'state': state_after_retrieval
        }
        
        # Step 2: Generate answer with streaming
        yield {
            'step': 'generating', 
            'content': '', 
            'state': state_after_retrieval
        }
        
        # Stream the answer generation
        final_state = None
        for answer_chunk in self.nodes.generate_answer_streaming(state_after_retrieval):
            if isinstance(answer_chunk, str):
                # This is a streaming chunk
                yield {
                    'step': 'generating',
                    'content': answer_chunk,
                    'state': state_after_retrieval
                }
            else:
                # This is the final state
                final_state = answer_chunk
        
        # Step 3: Complete
        if final_state:
            yield {
                'step': 'complete',
                'content': final_state.answer,
                'state': final_state
            }