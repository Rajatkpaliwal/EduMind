from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    Initialize and return HuggingFace sentence-transformer embeddings.
    Used for encoding document chunks into vector representations for RAG.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings

    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {str(e)}")