import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from utils.web_search import web_search
from utils.rag import retrieve_docs


def create_agent(llm, db):
    """
    Create a LangChain ZERO_SHOT_REACT agent with two tools:
      - Document Search (RAG over uploaded files)
      - Web Search (live internet search via Serper)

    Args:
        llm:  An initialized LangChain-compatible LLM.
        db:   FAISS vectorstore (or None if no documents are uploaded).

    Returns:
        A configured LangChain agent executor.
    """
    try:
        tools = [
            Tool(
                name="Document Search",
                func=lambda q: retrieve_docs(q, db),
                description=(
                    "Use this tool to search through uploaded documents or PDFs. "
                    "Useful when the user asks questions about specific files they have provided."
                ),
            ),
            Tool(
                name="Web Search",
                func=web_search,
                description=(
                    "Use this tool to search the internet for current events, recent news, "
                    "or any information not found in uploaded documents."
                ),
            ),
        ]

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
        )

        return agent

    except Exception as e:
        raise RuntimeError(f"Failed to create agent: {str(e)}")