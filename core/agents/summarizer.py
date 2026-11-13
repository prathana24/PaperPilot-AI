"""
Agent 1: Summarizer Agent
Generates a structured, multi-perspective summary of research papers
using ChromaDB retrieved context, ChatPromptTemplate, and LCEL.
Supports seamless offline extractive fallback when Gemini API is unconfigured.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_llm
from core.vectorstore import retrieve_relevant_passages
from core.offline_engine import generate_offline_extractive_summary

SUMMARIZER_SYSTEM_PROMPT = """You are an expert academic research assistant specializing in scientific literature synthesis.
Your mission is to provide an accurate, structured, and student-friendly summary of the research paper based ONLY on the provided retrieved excerpts.

STRICT GROUNDING RULES:
1. Ground all claims, metrics, and details strictly in the provided paper context.
2. If certain sections (such as limitations or specific baselines) are not discussed in the context, explicitly state "Not explicitly identified in the extracted text." rather than speculating.
3. Keep the tone academic, clear, and easy for university students to comprehend.

Structure your response using these exact markdown headers:
## 🎯 Research Objectives & Motivation
(What core problem does the paper solve, and why is it important?)

## 🧩 Problem Statement & Research Gaps
(What limitations in existing works are addressed?)

## ⚙️ Core Methodology & Proposed Approach
(Step-by-step explanation of the proposed system, algorithm, or framework)

## 📊 Major Findings & Quantitative Results
(Empirical performance, benchmarks, metrics, or experimental validation)

## ⚠️ Acknowledged Limitations & Constraints
(What boundaries, assumptions, or limitations are noted?)

## 🏁 Summary & Conclusion
(High-level takeaway and real-world significance)
"""

SUMMARIZER_HUMAN_PROMPT = """Document: {filename}

RETRIEVED PAPER EXCERPTS:
{context}

USER REQUEST:
{user_query}

Please produce the structured academic summary according to the guidelines."""


def create_summarizer_chain(llm):
    """Builds the LCEL chain: prompt | llm | StrOutputParser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARIZER_SYSTEM_PROMPT),
        ("human", SUMMARIZER_HUMAN_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


def run_summarizer_agent(
    vectorstore,
    filename: str,
    user_query: str = "Generate a comprehensive structured summary of this paper.",
    api_key: Optional[str] = None,
    full_text: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Executes the Summarizer Agent.
    
    If Gemini API key is available: runs the LangChain LCEL pipeline.
    If Gemini API key is unconfigured: falls back to the structured offline extractive engine.
    """
    llm, err = get_llm(api_key=api_key)

    # ---------------------------------------------------------
    # 1. OFFLINE FALLBACK PATH (When API key is not configured)
    # ---------------------------------------------------------
    if err or llm is None:
        # Retrieve context passages to serve as citations
        retrieval_query = f"{user_query} objective methodology results conclusion"
        retrieved_docs = retrieve_relevant_passages(retrieval_query, vectorstore, top_k=top_k) if vectorstore else []
        sources = [{"page": d["page"], "source": d["source"], "score": d["score"]} for d in retrieved_docs]

        # Use full_text if provided, or concatenate retrieved chunks
        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_summary = generate_offline_extractive_summary(context_text)

        return {
            "status": "success",
            "output": offline_summary,
            "sources": sources,
            "engine": "offline_extractive",
            "label": "Offline Extractive Summary — No Gemini API"
        }

    # ---------------------------------------------------------
    # 2. LIVE GEMINI LCEL PATH (When API key is configured)
    # ---------------------------------------------------------
    retrieval_query = f"{user_query} objective methodology results conclusion"
    retrieved_docs = retrieve_relevant_passages(retrieval_query, vectorstore, top_k=top_k) if vectorstore else []

    if not retrieved_docs:
        return {
            "status": "error",
            "error": "No relevant context could be retrieved from the document vector store.",
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
        chain = create_summarizer_chain(llm)
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
            "label": "Gemini-Powered Structured Summary (LCEL)"
        }
    except Exception as e:
        # If API invocation fails unexpectedly, fallback gracefully to offline extraction
        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_summary = generate_offline_extractive_summary(context_text)
        return {
            "status": "success",
            "output": f"> ⚠️ **Gemini API Call Failed:** `{type(e).__name__}`. Showing offline extraction.\n\n" + offline_summary,
            "sources": sources,
            "engine": "offline_extractive_fallback",
            "label": "Offline Extractive Summary — API Fallback"
        }
