import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, MODEL_NAME


def get_chatgroq_model():
    """
    Initialize and return the Groq chat model.
    Raises RuntimeError if initialization fails.
    """
    try:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file.")

        groq_model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME,
        )
        return groq_model

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")