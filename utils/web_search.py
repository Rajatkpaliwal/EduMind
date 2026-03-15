import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.utilities import GoogleSerperAPIWrapper
from config.config import SERPER_API_KEY

def web_search(query: str) -> str:
    """
    Perform a real-time web search using the Google Serper API.
    Returns a string with the search results, or an error message.
    """
    try:
        if not SERPER_API_KEY:
            return "Web search is disabled: SERPER_API_KEY is not configured."

        os.environ["SERPER_API_KEY"] = SERPER_API_KEY

        search = GoogleSerperAPIWrapper()
        results = search.run(query)

        if not results:
            return "No web search results found."

        return results

    except Exception as e:
        return f"Web search error: {str(e)}"