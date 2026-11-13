"""
Intent Detection, Agent Router, and Grounded QA Pipeline for PaperPilot-AI.
Classifies user intent, dispatches requests to the 4 specialized agents or
the Grounded QA engine, and returns standardized execution payloads.
Supports seamless offline extractive fallbacks across all routes.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.config import get_llm
from core.vectorstore import retrieve_relevant_passages
from core.memory import ConversationMemory
from core.agents.summarizer import run_summarizer_agent
from core.agents.concept_extractor import run_concept_extractor_agent
from core.agents.slide_generator import run_slide_generator_agent
from core.agents.viva_agent import run_viva_agent

# Supported Intent Categories
INTENT_SUMMARIZE = "summarize"
INTENT_CONCEPTS = "extract_concepts"
INTENT_SLIDES = "generate_slides"
INTENT_VIVA = "prepare_viva"
INTENT_QA = "ask_question"
INTENT_AMBIGUOUS = "ambiguous"

INTENT_MAP = {
    INTENT_SUMMARIZE: "Summarizer Agent",
    INTENT_CONCEPTS: "Concept-Extractor Agent",
    INTENT_SLIDES: "Slide-Generator Agent",
    INTENT_VIVA: "Viva Agent",
    INTENT_QA: "Grounded Q&A Engine",
    INTENT_AMBIGUOUS: "Disambiguation Handler",
}

# -----------------------------------------------------------------------------
# Grounded Q&A Chain (For INTENT_QA)
# -----------------------------------------------------------------------------
QA_SYSTEM_PROMPT = """You are PaperPilot-AI's Grounded Question Answering Engine.
Your objective is to answer user queries accurately based EXCLUSIVELY on the retrieved research paper context.

RULES:
1. Ground your answer strictly in the provided paper context.
2. If the answer cannot be determined from the excerpts, explicitly state:
   "Not explicitly identified in the extracted text."
3. Cite the relevant page numbers whenever discussing specific points or metrics.
4. If relevant, take previous conversation context into account.
"""

QA_HUMAN_PROMPT = """Document: {filename}

CONVERSATION HISTORY:
{chat_history}

RETRIEVED PAPER EXCERPTS:
{context}

USER QUESTION:
{question}

Provide an accurate, grounded answer:"""


def create_grounded_qa_chain(llm):
    """Builds the LCEL chain for grounded question answering."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("human", QA_HUMAN_PROMPT),
    ])
    return prompt | llm | StrOutputParser()


# -----------------------------------------------------------------------------
# Rule-Based Intent Classifier (Deterministic Fallback)
# -----------------------------------------------------------------------------
RULE_PATTERNS = {
    INTENT_SUMMARIZE: [
        r"\b(?:summarize|summary|overview|tldr|brief|abstract|synopsis|executive\s+summary)\b",
        r"\b(?:what\s+is\s+this\s+paper\s+about|give\s+me\s+a\s+summary)\b",
    ],
    INTENT_CONCEPTS: [
        r"\b(?:concepts?|methodology|methods?|algorithms?|architecture|formulas?|techniques?|math|theoretical)\b",
        r"\b(?:explain\s+how\s+it\s+works|extract\s+concepts|break\s+down\s+the\s+approach)\b",
    ],
    INTENT_SLIDES: [
        r"\b(?:slides?|presentation|deck|powerpoint|pptx|keynote|bullet\s+points\s+for\s+talk)\b",
        r"\b(?:create\s+slides|generate\s+presentation|make\s+a\s+deck)\b",
    ],
    INTENT_VIVA: [
        r"\b(?:viva|exam|defense|oral\s+questions?|interview\s+questions?|quiz|test\s+me)\b",
        r"\b(?:questions?\s+for\s+defense|prepare\s+for\s+viva|viva\s+voce)\b",
    ],
}


def classify_intent_rule_based(query: str) -> str:
    """Classifies user intent using deterministic keyword and regex heuristics."""
    q_lower = query.lower().strip()
    
    if not q_lower:
        return INTENT_AMBIGUOUS

    # Match count across categories
    scores: Dict[str, int] = {}
    for intent, patterns in RULE_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, q_lower))
        if score > 0:
            scores[intent] = score

    # Check for ambiguity (multiple strong distinct intent matches)
    if len(scores) > 1:
        max_score = max(scores.values())
        top_intents = [k for k, v in scores.items() if v == max_score]
        if len(top_intents) > 1:
            return INTENT_AMBIGUOUS
        return top_intents[0]

    if len(scores) == 1:
        return list(scores.keys())[0]

    # Specific question indicators default to Q&A
    if any(q_lower.startswith(w) for w in ["what", "why", "how", "when", "where", "who", "which", "is", "can", "does", "did", "are"]):
        return INTENT_QA

    # If the user asks a descriptive query with at least 3 words, treat as Q&A
    words = q_lower.split()
    if len(words) >= 3:
        return INTENT_QA

    return INTENT_AMBIGUOUS


# -----------------------------------------------------------------------------
# LLM-Based Intent Classifier (When Gemini is active)
# -----------------------------------------------------------------------------
INTENT_CLASSIFIER_PROMPT = """You are an intent classification component for PaperPilot-AI, an academic research assistant.
Analyze the user's input and classify it into EXACTLY ONE of the following categories:

- summarize (if user wants a summary, overview, or general recap of the paper)
- extract_concepts (if user asks about methods, algorithms, architecture, formulas, or concepts)
- generate_slides (if user wants presentation slides, slide deck, or talk outlines)
- prepare_viva (if user asks for viva questions, defense prep, oral exam questions, or practice)
- ask_question (if user asks a specific factual, analytical, or comparative question about the paper)
- ambiguous (if the query is empty, purely greeting like 'hi', gibberish, or unclear)

Respond with ONLY the category name in lowercase (e.g., 'summarize' or 'ask_question'). No punctuation, no extra words."""


def classify_intent(query: str, api_key: Optional[str] = None) -> Tuple[str, str]:
    """Classifies user intent using Gemini when available, falling back to rule-based classification."""
    llm, err = get_llm(api_key=api_key, temperature=0.0, max_output_tokens=30)
    
    if llm is not None and not err:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", INTENT_CLASSIFIER_PROMPT),
                ("human", "{query}"),
            ])
            chain = prompt | llm | StrOutputParser()
            res = chain.invoke({"query": query}).strip().lower()
            cleaned_intent = re.sub(r"[^a-z_]", "", res)
            if cleaned_intent in INTENT_MAP:
                return cleaned_intent, "gemini_classifier"
        except Exception:
            pass  # Fall back to rule-based

    return classify_intent_rule_based(query), "rule_based_fallback"


# -----------------------------------------------------------------------------
# Agent Router & Execution Pipeline
# -----------------------------------------------------------------------------
def route_and_execute(
    user_query: str,
    vectorstore,
    filename: str,
    memory: Optional[ConversationMemory] = None,
    api_key: Optional[str] = None,
    full_text: Optional[str] = None,
    explicit_intent: Optional[str] = None,
) -> Dict[str, Any]:
    """Routes the user request to the designated agent and executes the pipeline."""
    if explicit_intent and explicit_intent in INTENT_MAP:
        intent = explicit_intent
        classifier_used = "explicit_override"
    else:
        intent, classifier_used = classify_intent(user_query, api_key=api_key)

    selected_agent = INTENT_MAP.get(intent, "Unknown Agent")

    # Handle Ambiguous Intent
    if intent == INTENT_AMBIGUOUS:
        msg = (
            "I could not clearly determine your objective. Would you like me to:\n"
            "1. 📝 **Summarize** this research paper?\n"
            "2. 💡 **Extract Core Concepts** & methodology?\n"
            "3. 🖥️ **Generate Presentation Slides**?\n"
            "4. 🎓 **Prepare 10 Viva Exam Questions**?\n"
            "5. 💬 **Answer a specific question** about the document?\n\n"
            "Please rephrase your request or choose an option above."
        )
        if memory:
            memory.add_user_message(user_query)
            memory.add_assistant_message(msg, intent=intent, agent=selected_agent)
        return {
            "status": "ambiguous",
            "intent": intent,
            "agent": selected_agent,
            "output": msg,
            "sources": [],
            "classifier_engine": classifier_used,
            "engine": "disambiguation_handler",
        }

    # Dispatch to Specialized Agent
    if intent == INTENT_SUMMARIZE:
        result = run_summarizer_agent(vectorstore, filename, user_query=user_query, api_key=api_key, full_text=full_text)
    elif intent == INTENT_CONCEPTS:
        result = run_concept_extractor_agent(vectorstore, filename, user_query=user_query, api_key=api_key, full_text=full_text)
    elif intent == INTENT_SLIDES:
        result = run_slide_generator_agent(vectorstore, filename, user_query=user_query, api_key=api_key, full_text=full_text)
    elif intent == INTENT_VIVA:
        result = run_viva_agent(vectorstore, filename, user_query=user_query, api_key=api_key, full_text=full_text)
    elif intent == INTENT_QA:
        llm, err = get_llm(api_key=api_key)
        retrieved = retrieve_relevant_passages(user_query, vectorstore, top_k=4) if vectorstore else []
        sources = [{"page": r["page"], "source": r["source"], "score": r["score"]} for r in retrieved]

        if err or llm is None:
            # Offline Extractive Q&A Fallback
            if retrieved:
                bullets = "\n\n".join([f"• **[Page {r['page']}]**: {r['content']}" for r in retrieved])
                extractive_msg = (
                    f"**Grounded Passages Found in Document (Offline Extractive Retrieval):**\n\n{bullets}\n\n"
                    "*(Notice: Retrieved directly from ChromaDB vector similarity without external API calls.)*"
                )
                result = {
                    "status": "success",
                    "output": extractive_msg,
                    "sources": sources,
                    "engine": "offline_extractive",
                    "label": "Grounded Passage Retrieval — No Gemini API"
                }
            else:
                result = {
                    "status": "success",
                    "output": "Not explicitly identified in the extracted text.",
                    "sources": [],
                    "engine": "offline_extractive",
                    "label": "Not Found"
                }
        else:
            # Live LLM Grounded Q&A
            formatted_context = "\n\n---\n\n".join([f"[Page {r['page']}]: {r['content']}" for r in retrieved])
            history_text = memory.get_formatted_history(limit=4) if memory else "No previous turns."

            try:
                qa_chain = create_grounded_qa_chain(llm)
                answer = qa_chain.invoke({
                    "filename": filename,
                    "chat_history": history_text,
                    "context": formatted_context,
                    "question": user_query,
                })
                result = {
                    "status": "success",
                    "output": answer,
                    "sources": sources,
                    "engine": "langchain_lcel_gemini",
                    "label": "Gemini-Powered Grounded Q&A (LCEL)"
                }
            except Exception as e:
                # Graceful fallback to retrieved passages
                bullets = "\n\n".join([f"• **[Page {r['page']}]**: {r['content']}" for r in retrieved])
                result = {
                    "status": "success",
                    "output": f"> ⚠️ **Gemini API Call Failed:** `{type(e).__name__}`. Showing grounded passages.\n\n" + bullets,
                    "sources": sources,
                    "engine": "offline_extractive_fallback",
                    "label": "Grounded Passage Retrieval — API Fallback"
                }
    else:
        result = {
            "status": "error",
            "error": f"Unknown intent: {intent}",
            "output": None,
            "sources": [],
            "engine": "unknown"
        }

    # Record in Memory
    if memory and result.get("output"):
        memory.add_user_message(user_query)
        memory.add_assistant_message(
            content=result["output"],
            intent=intent,
            agent=selected_agent,
            sources=result.get("sources", []),
        )

    result["intent"] = intent
    result["agent"] = selected_agent
    result["classifier_engine"] = classifier_used

    return result
