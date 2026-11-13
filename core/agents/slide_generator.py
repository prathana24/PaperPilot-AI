"""
Agent 3: Slide-Generator Agent
Transforms research paper findings into a structured 6-8 slide presentation deck
with slide titles, concise bullet points, and presenter speaking notes using LCEL.
Supports seamless offline fallback when Gemini API is unconfigured.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_llm
from core.vectorstore import retrieve_relevant_passages
from core.offline_engine import generate_offline_slides

SLIDE_SYSTEM_PROMPT = """You are an academic presentation design specialist.
Your goal is to convert the core findings of a scientific research paper into a structured, highly professional 6 to 8 slide presentation deck.

GUIDELINES:
1. Generate between 6 and 8 clearly numbered slides.
2. Each slide must contain:
   - Slide Number and Title
   - 3 to 5 concise, impactful bullet points
   - "🎙️ Speaker Notes": A realistic spoken narrative explaining the slide content to an academic audience.
3. Standard Slide Progression:
   - Slide 1: Title & Overview
   - Slide 2: Problem Formulation & Motivation
   - Slide 3: Proposed Methodology & Architecture
   - Slide 4: Key Algorithmic / Technical Innovations
   - Slide 5: Experimental Setup & Benchmarks
   - Slide 6: Results & Comparative Analysis
   - Slide 7: Critical Discussion & Limitations
   - Slide 8: Conclusion & Future Outlook
4. Ground all slide bullet points strictly in the provided paper context.

FORMATTING SPECIFICATION:
Format each slide cleanly in Markdown as follows:
---
### Slide X: [Slide Title]
- [Bullet 1]
- [Bullet 2]
- [Bullet 3]

**🎙️ Speaker Notes:**
[Spoken script for the presenter]
---
"""

SLIDE_HUMAN_PROMPT = """Document: {filename}

RETRIEVED PAPER EXCERPTS:
{context}

SLIDE CUSTOMIZATION PREFERENCES:
{user_query}

Create the complete 6 to 8 slide presentation deck now."""


def create_slide_generator_chain(llm):
    """Builds the LCEL chain: prompt | llm | StrOutputParser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SLIDE_SYSTEM_PROMPT),
        ("human", SLIDE_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


def run_slide_generator_agent(
    vectorstore,
    filename: str,
    user_query: str = "Generate an 8-slide conference presentation deck with speaker notes.",
    api_key: Optional[str] = None,
    full_text: Optional[str] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Executes the Slide-Generator Agent with live LCEL or offline generation."""
    llm, err = get_llm(api_key=api_key)

    # 1. OFFLINE FALLBACK PATH
    if err or llm is None:
        retrieval_query = f"{user_query} problem architecture results conclusion methodology"
        retrieved_docs = retrieve_relevant_passages(retrieval_query, vectorstore, top_k=top_k) if vectorstore else []
        sources = [{"page": d["page"], "source": d["source"], "score": d["score"]} for d in retrieved_docs]

        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_slides = generate_offline_slides(context_text, filename)

        return {
            "status": "success",
            "output": offline_slides,
            "sources": sources,
            "engine": "offline_extractive",
            "label": "Offline Slide Outline — No Gemini API"
        }

    # 2. LIVE GEMINI LCEL PATH
    retrieval_query = f"{user_query} problem architecture results conclusion methodology"
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
        chain = create_slide_generator_chain(llm)
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
            "label": "Gemini-Powered Slide Deck (LCEL)"
        }
    except Exception as e:
        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_slides = generate_offline_slides(context_text, filename)
        return {
            "status": "success",
            "output": f"> ⚠️ **Gemini API Call Failed:** `{type(e).__name__}`. Showing offline slides.\n\n" + offline_slides,
            "sources": sources,
            "engine": "offline_extractive_fallback",
            "label": "Offline Slide Outline — API Fallback"
        }
