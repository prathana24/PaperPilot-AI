"""
Vector Store, Chunking, and Local Retrieval Pipeline for PaperPilot-AI.
Implements RecursiveCharacterTextSplitter, local HuggingFace embeddings,
and ChromaDB document storage with metadata preservation.
"""

import os
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Base directory for persistent ChromaDB storage
CHROMA_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")

# Singleton cache for HuggingFace embedding model to prevent redundant reloads
_EMBEDDINGS_INSTANCE: Optional[HuggingFaceEmbeddings] = None


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Initializes and caches the local HuggingFace embedding model.
    
    Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional).
    Runs completely locally on CPU/GPU without external API quota consumption.
    """
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        try:
            _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize HuggingFace embeddings: {str(e)}. "
                "Ensure sentence-transformers is installed and models can be loaded."
            )
    return _EMBEDDINGS_INSTANCE


def chunk_document_pages(
    pages_dict: Dict[int, str],
    source_filename: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Splits per-page text into overlapping chunks using RecursiveCharacterTextSplitter.
    
    Preserves source document name, page number, and chunk index in metadata.
    """
    if not pages_dict:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: List[Document] = []
    chunk_idx = 0

    for page_num in sorted(pages_dict.keys()):
        page_text = pages_dict[page_num]
        if not page_text or not page_text.strip():
            continue

        raw_chunks = splitter.split_text(page_text)
        for text_chunk in raw_chunks:
            cleaned = text_chunk.strip()
            if len(cleaned) < 15:
                continue

            metadata = {
                "source": source_filename,
                "page": page_num,
                "chunk_id": chunk_idx,
            }
            all_chunks.append(Document(page_content=cleaned, metadata=metadata))
            chunk_idx += 1

    return all_chunks


def get_collection_name(filename: str) -> str:
    """Generates a valid, sanitized ChromaDB collection name unique to the document."""
    # Chroma requires collection names to be 3-63 characters: alphanumeric, underscores, or hyphens
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", filename)
    name_hash = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    clean_name = f"paper_{sanitized[:40]}_{name_hash}"
    # Ensure start and end with alphanumeric
    clean_name = re.sub(r"^[^a-zA-Z0-9]+", "doc_", clean_name)
    clean_name = re.sub(r"[^a-zA-Z0-9]+$", "", clean_name)
    return clean_name[:63]


def index_paper_chunks(
    chunks: List[Document],
    filename: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Indexes document chunks into a persistent, isolated Chroma collection.
    
    Handles deduplication and prevents cross-document pollution.
    """
    if not chunks:
        raise ValueError("Cannot index an empty list of chunks.")

    os.makedirs(persist_dir, exist_ok=True)
    embeddings = get_embeddings_model()
    collection_name = get_collection_name(filename)

    # Initialize or load Chroma vector store
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Check if this document collection already contains embeddings
    existing_count = vectorstore._collection.count()
    if existing_count > 0:
        # Document already indexed in this persistent collection
        return vectorstore

    # Ingest chunks
    vectorstore.add_documents(chunks)
    return vectorstore


def get_paper_vectorstore(
    filename: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
) -> Optional[Chroma]:
    """Retrieves an existing Chroma collection for a document if previously indexed."""
    collection_name = get_collection_name(filename)
    embeddings = get_embeddings_model()
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        if vectorstore._collection.count() > 0:
            return vectorstore
        return None
    except Exception:
        return None


def retrieve_relevant_passages(
    query: str,
    vectorstore: Chroma,
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Performs similarity search with relevance scoring for a given query.
    
    Returns:
        List of dicts with:
        - content: Chunk text
        - page: Page number in source document
        - source: Document filename
        - score: Similarity score (cosine distance)
    """
    if not query or not query.strip():
        return []

    if vectorstore is None:
        return []

    try:
        docs_and_scores = vectorstore.similarity_search_with_score(query.strip(), k=top_k)
        results = []
        for doc, score in docs_and_scores:
            results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 1),
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "score": float(score),
            })
        return results
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []
