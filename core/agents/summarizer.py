"""
Agent 1: Summarizer Agent
Generates a structured, multi-perspective summary of research papers
using ChromaDB retrieved context, ChatPromptTemplate, and LCEL.
Incorporates a two-step Reflection & Self-Correction loop:
  Step 1: Initial Drafting
  Step 2: Self-Critique & Grounding Verification
Supports seamless offline extractive fallback when Gemini API is unconfigured.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_llm
from core.vectorstore import retrieve_relevant_passages
from core.offline_engine import generate_offline_extractive_summary

# -----------------------------------------------------------------------------
# STEP 1: INITIAL DRAFTING PROMPTS
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# STEP 2: REFLECTION & SELF-CORRECTION PROMPTS
# -----------------------------------------------------------------------------
REFLECTION_SYSTEM_PROMPT = """You are a rigorous Academic Peer Reviewer and Fact-Checking Agent.
Your task is to audit and self-correct a proposed draft summary against the original retrieved paper excerpts.

AUDIT CRITERIA:
1. Factual Grounding: Are there any claims, statistics, or benchmarks in the draft summary that are NOT supported by the retrieved excerpts?
2. Hallucination Check: Did the draft invent missing sections, algorithms, or baseline results?
3. Completeness: Did the draft miss critical empirical percentages, throughput numbers, or stated limitations present in the excerpts?
4. Section Formatting: Does the draft strictly follow the required 6 markdown headers?

INSTRUCTIONS:
- If the draft is completely faithful to the paper excerpts and contains no hallucinations or missing empirical facts, return the draft summary unchanged.
- If any claim is ungrounded or exaggerated, correct it immediately to align strictly with the excerpts.
- Replace any speculative claims with: "Not explicitly identified in the extracted text."
- Return ONLY the final polished markdown summary. Do not add introductory conversational filler like 'Here is the revised summary'."""

REFLECTION_HUMAN_PROMPT = """ORIGINAL RETRIEVED PAPER EXCERPTS:
{context}

PROPOSED DRAFT SUMMARY TO AUDIT:
{draft_summary}

Audit and self-correct the summary now based on the original excerpts:"""


def create_summarizer_chain(llm):
    """Builds the initial drafting LCEL chain: prompt | llm | StrOutputParser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUMMARIZER_SYSTEM_PROMPT),
        ("human", SUMMARIZER_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


def create_reflection_chain(llm):
    """Builds the reflection/self-correction LCEL chain: prompt | llm | StrOutputParser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", REFLECTION_SYSTEM_PROMPT),
        ("human", REFLECTION_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


def run_summarizer_agent(
    vectorstore,
    filename: str,
    user_query: str = "Generate a comprehensive structured summary of this paper.",
    api_key: Optional[str] = None,
    full_text: Optional[str] = None,
    top_k: int = 5,
    enable_reflection: bool = True,
) -> Dict[str, Any]:
    """Executes the Summarizer Agent with a Reflection & Self-Correction step.
    
    If Gemini API key is available:
      1. Generates initial draft summary using LCEL.
      2. If enable_reflection is True, passes the draft and source context to the Reflection Agent
         to audit factual grounding, eliminate hallucinations, and output a verified final summary.
    If Gemini API key is unconfigured: falls back to the structured offline extractive engine.
    """
    llm, err = get_llm(api_key=api_key)

    # ---------------------------------------------------------
    # 1. OFFLINE FALLBACK PATH (When API key is not configured)
    # ---------------------------------------------------------
    if err or llm is None:
        retrieval_query = f"{user_query} objective methodology results conclusion"
        retrieved_docs = retrieve_relevant_passages(retrieval_query, vectorstore, top_k=top_k) if vectorstore else []
        sources = [{"page": d["page"], "source": d["source"], "score": d["score"]} for d in retrieved_docs]

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
    # 2. LIVE GEMINI LCEL PATH WITH REFLECTION & SELF-CORRECTION
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
        # Step 1: Initial Drafting
        drafting_chain = create_summarizer_chain(llm)
        draft_summary = drafting_chain.invoke({
            "filename": filename,
            "context": formatted_context,
            "user_query": user_query,
        })

        final_summary = draft_summary

        # Step 2: Reflection & Self-Correction (if enabled)
        if enable_reflection:
            reflection_chain = create_reflection_chain(llm)
            reflected_summary = reflection_chain.invoke({
                "context": formatted_context,
                "draft_summary": draft_summary,
            })
            if reflected_summary and len(reflected_summary.strip()) > 100:
                final_summary = reflected_summary

        return {
            "status": "success",
            "output": final_summary,
            "sources": sources,
            "engine": "langchain_lcel_gemini_reflection",
            "label": "Gemini-Powered Structured Summary with Reflection (LCEL)"
        }
    except Exception as e:
        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_summary = generate_offline_extractive_summary(context_text)
        return {
            "status": "success",
            "output": f"> ⚠️ **Gemini API Call Failed:** `{type(e).__name__}`. Showing offline extraction.\n\n" + offline_summary,
            "sources": sources,
            "engine": "offline_extractive_fallback",
            "label": "Offline Extractive Summary — API Fallback"
        }
