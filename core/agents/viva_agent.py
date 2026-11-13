"""
Agent 4: Viva Agent
Generates 10 research-paper-specific viva-voce questions spanning basic,
intermediate, and advanced difficulty with comprehensive model answers using LCEL.
Supports seamless offline fallback when Gemini API is unconfigured.
"""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_llm
from core.vectorstore import retrieve_relevant_passages
from core.offline_engine import generate_offline_viva

VIVA_SYSTEM_PROMPT = """You are a senior university professor and oral thesis examination committee member.
Your task is to generate exactly 10 rigorous, highly relevant viva-voce examination questions with comprehensive model answers based on the provided research paper.

QUESTION DIFFICULTY SPECTRUM:
- Questions 1 to 3 (Foundational / Basic): Motivations, problem definition, core terminology, scope.
- Questions 4 to 7 (Intermediate / Architectural): Algorithmic mechanisms, architectural design choices, baseline comparisons, experimental validation.
- Questions 8 to 10 (Advanced / Critical Defense): Trade-offs, failure modes, scalability constraints, assumptions, and critical theoretical defense.

STRICT GROUNDING RULES:
1. Ground questions and model answers strictly in the paper context.
2. In each model answer, highlight the specific technical rationale presented by the authors.
3. Include an "Examiner Focus Tip" explaining what examiners look for during the oral defense.

FORMAT SPECIFICATION:
### Question X: [Clear, targeted oral defense question]
- **Difficulty Level:** [Basic / Intermediate / Advanced]
- **Ideal Model Answer:** [Clear, authoritative answer grounded in the paper]
- **💡 Examiner Focus Tip:** [What key insight or depth the candidate must demonstrate]
---
"""

VIVA_HUMAN_PROMPT = """Document: {filename}

RETRIEVED PAPER EXCERPTS:
{context}

CUSTOM EXAM PREFERENCES:
{user_query}

Generate the 10 university viva questions and answers now."""


def create_viva_chain(llm):
    """Builds the LCEL chain: prompt | llm | StrOutputParser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", VIVA_SYSTEM_PROMPT),
        ("human", VIVA_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


def run_viva_agent(
    vectorstore,
    filename: str,
    user_query: str = "Generate 10 comprehensive viva-voce questions and model answers.",
    api_key: Optional[str] = None,
    full_text: Optional[str] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Executes the Viva Agent with live LCEL or offline generation."""
    llm, err = get_llm(api_key=api_key)

    # 1. OFFLINE FALLBACK PATH
    if err or llm is None:
        retrieval_query = f"{user_query} limitations motivation methodology experimental results conclusion"
        retrieved_docs = retrieve_relevant_passages(retrieval_query, vectorstore, top_k=top_k) if vectorstore else []
        sources = [{"page": d["page"], "source": d["source"], "score": d["score"]} for d in retrieved_docs]

        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_viva = generate_offline_viva(context_text)

        return {
            "status": "success",
            "output": offline_viva,
            "sources": sources,
            "engine": "offline_extractive",
            "label": "Offline Viva Questions — No Gemini API"
        }

    # 2. LIVE GEMINI LCEL PATH
    retrieval_query = f"{user_query} limitations motivation methodology experimental results conclusion"
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
        chain = create_viva_chain(llm)
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
            "label": "Gemini-Powered Viva Examination (LCEL)"
        }
    except Exception as e:
        context_text = full_text if full_text else " ".join([d["content"] for d in retrieved_docs])
        offline_viva = generate_offline_viva(context_text)
        return {
            "status": "success",
            "output": f"> ⚠️ **Gemini API Call Failed:** `{type(e).__name__}`. Showing offline questions.\n\n" + offline_viva,
            "sources": sources,
            "engine": "offline_extractive_fallback",
            "label": "Offline Viva Questions — API Fallback"
        }
