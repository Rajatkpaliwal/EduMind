import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embeddings
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, VECTORSTORE_DIR


def load_documents(file_path: str):
    """
    Load documents from a PDF or TXT file.
    Returns a list of LangChain Document objects.
    """
    try:
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .txt are supported.")

        documents = loader.load()
        return documents

    except Exception as e:
        raise RuntimeError(f"Error loading document '{file_path}': {str(e)}")


def split_documents(documents):
    """
    Split documents into overlapping chunks for embedding.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        return chunks

    except Exception as e:
        raise RuntimeError(f"Error splitting documents: {str(e)}")


def build_vectorstore(file_paths: list):
    """
    Build a FAISS vector store from a list of uploaded file paths.
    Returns the FAISS vectorstore object.
    """
    try:
        all_chunks = []

        for file_path in file_paths:
            docs = load_documents(file_path)
            chunks = split_documents(docs)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No content could be extracted from the uploaded files.")

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        return vectorstore

    except Exception as e:
        raise RuntimeError(f"Error building vector store: {str(e)}")


def retrieve_docs(query: str, db) -> str:
    """
    Retrieve the top-K most relevant document chunks for a given query.
    Returns a formatted string of the retrieved context.
    """
    try:
        if db is None:
            return "No documents uploaded. Please upload documents to enable document search."

        results = db.similarity_search(query, k=TOP_K_RESULTS)

        if not results:
            return "No relevant content found in the uploaded documents."

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page + 1})" if page != "" else ""
            context_parts.append(f"[Excerpt {i} from {os.path.basename(source)}{page_info}]:\n{doc.page_content}")

        return "\n\n".join(context_parts)

    except Exception as e:
        return f"Document retrieval error: {str(e)}"