"""
Agent 2: Concept-Extractor Agent
Extracts core concepts, mathematical formulations, algorithms, and empirical metrics
from research papers using ChromaDB retrieval and LCEL.
Supports seamless offline fallback when Gemini API is unconfigured.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_llm
from core.vectorstore import retrieve_relevant_passages
from core.offline_engine import generate_offline_concepts

CONCEPT_SYSTEM_PROMPT = """You are a specialized Computer Science and Machine Learning Concept Extraction Agent.
Your role is to dissect research papers into clear, modular technical concepts and explain them in intuitive, plain language without sacrificing scientific rigor.

STRICT INSTRUCTIONS:
1. Extract the key theoretical and practical concepts explicitly mentioned in the context.
2. Explain complex algorithms, architectures, or formulas in plain, easy-to-understand English.
3. Extract precise numerical results, percentages, latency numbers, and benchmarks IF present in the context. NEVER fabricate numbers or benchmarks.
4. If equations or algorithms are mentioned, describe their inputs, functional steps, and outputs.

Structure your output under the following sections:
## 💡 Core Theoretical Concepts
- Breakdown of fundamental ideas and terminology introduced or utilized.

## 🛠️ Algorithms & System Architecture
- Detailed explanation of how the system works mechanically, step-by-step.

## 📈 Quantitative Benchmarks & Empirical Metrics
- Precise figures, percentages, accuracy scores, or speed improvements explicitly reported.
- Mention "Not explicitly identified in the extracted text." if absent.

## 🎓 Student Explanation (ELI5)
- An intuitive analogy or simple explanation explaining the primary contribution to a beginner undergraduate.
"""

CONCEPT_HUMAN_PROMPT = """Document: {filename}

RETRIEVED EXCERPTS:
{context}

FOCUS TOPIC / USER PROMPT:
{user_query}

Extract and break down the core concepts following the required structure."""


def create_concept_extractor_chain(llm):
    """Builds the LCEL chain: prompt | llm | StrOutputParser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONCEPT_SYSTEM_PROMPT),
        ("human", CONCEPT_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


def run_concept_extractor_agent(
    vectorstore,
    filename: str,
    user_query: str = "Extract and explain all key concepts, methods, and quantitative results.",
    api_key: Optional[str] = None,
    full_text: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Executes the Concept-Extractor Agent with live LCEL or offline extraction."""
    llm, err = get_llm(api_key=api_key)

    # 1. OFFLINE FALLBACK PATH
    if err or llm is None:
        retrieval_query = f"{user_query} algorithm framework architecture results metrics"
        retrieved_docs = retrieve_relevant_passages(retrieval_query, vectorstore, top_k=top_k) if vectorstore else []
        sources = [{"page": d["page"], "source": d["source"], "score": d["score"]} for d in retrieved_docs]

        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_concepts = generate_offline_concepts(context_text)

        return {
            "status": "success",
            "output": offline_concepts,
            "sources": sources,
            "engine": "offline_extractive",
            "label": "Offline Extractive Concepts — No Gemini API"
        }

    # 2. LIVE GEMINI LCEL PATH
    retrieval_query = f"{user_query} algorithm framework architecture results metrics"
    retrieved_docs = retrieve_relevant_passages(retrieval_query, vectorstore, top_k=top_k) if vectorstore else []

    if not retrieved_docs:
        return {
            "status": "error",
            "error": "No relevant context found in vector store.",
            "output": None,
            "sources": [],
            "engine": "chroma_retrieval_failed",
            "label": "Error"
        }

    formatted_context = "\n\n---\n\n".join([
        f"[Excerpt from Page {d['page']}]:\n{d['content']}"
        for d in retrieved_docs
    ])
    sources = [{"page": d["page"], "source": d["source"], "score": d["score"]} for d in retrieved_docs]

    try:
        chain = create_concept_extractor_chain(llm)
        result = chain.invoke({
            "filename": filename,
            "context": formatted_context,
            "user_query": user_query,
        })
        return {
            "status": "success",
            "output": result,
            "sources": sources,
            "engine": "langchain_lcel_gemini",
            "label": "Gemini-Powered Concept Extraction (LCEL)"
        }
    except Exception as e:
        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_concepts = generate_offline_concepts(context_text)
        return {
            "status": "success",
            "output": f"> ⚠️ **Gemini API Call Failed:** `{type(e).__name__}`. Showing offline extraction.\n\n" + offline_concepts,
            "sources": sources,
            "engine": "offline_extractive_fallback",
            "label": "Offline Extractive Concepts — API Fallback"
        }
