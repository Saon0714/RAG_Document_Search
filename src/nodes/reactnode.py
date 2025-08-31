"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content with Streaming"""

from typing import List, Optional, Generator
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed corpus.",
            func=retriever_tool_fn,
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wiki.run,
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia.
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )

    def generate_answer_streaming(self, state: RAGState) -> Generator[str, None, RAGState]:
        """
        Generate answer with streaming support using ReAct agent.
        
        Args:
            state: Current RAG state with retrieved documents
            
        Yields:
            Partial answer chunks as they are generated
            
        Returns:
            Final RAGState with complete answer
        """
        if self._agent is None:
            self._build_agent()

        try:
            # Stream the agent response
            full_answer = ""
            for chunk in self._agent.stream({"messages": [HumanMessage(content=state.question)]}):
                # Extract the latest message content from the chunk
                if 'messages' in chunk and chunk['messages']:
                    latest_message = chunk['messages'][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        # For streaming, we get incremental content
                        if hasattr(latest_message, 'response_metadata') and latest_message.response_metadata.get('finish_reason'):
                            # This is the final message
                            full_answer = latest_message.content
                            yield full_answer
                        else:
                            # This is a partial message - yield the content
                            yield latest_message.content
                            full_answer = latest_message.content

            # If we didn't get a proper answer through streaming, fall back to invoke
            if not full_answer:
                result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
                messages = result.get("messages", [])
                if messages:
                    answer_msg = messages[-1]
                    full_answer = getattr(answer_msg, "content", "Could not generate answer.")
                    yield full_answer

            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=full_answer
            )

        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            yield error_msg
            return RAGState(
                question=state.question,
                retrieved_docs=state.retrieved_docs,
                answer=error_msg
            )