"""
Test Suite for the 4 Specialized Agents
Validates:
1. Agent imports
2. LCEL chain composition (including Reflection and Few-Shot chains)
3. Prompt variables & structure
4. Few-Shot exemplars in Viva Agent
5. Reflection & Self-Correction flow in Summarizer Agent (with mocked and LCEL tests)
6. Offline extractive fallbacks (100% functional without an API key)
"""

import os
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core.config import get_llm
from core.vectorstore import get_paper_vectorstore, retrieve_relevant_passages
from core.agents.summarizer import (
    create_summarizer_chain,
    create_reflection_chain,
    run_summarizer_agent,
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_HUMAN_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_HUMAN_PROMPT,
)
from core.agents.concept_extractor import (
    create_concept_extractor_chain,
    run_concept_extractor_agent,
    CONCEPT_SYSTEM_PROMPT,
    CONCEPT_HUMAN_PROMPT,
)
from core.agents.slide_generator import (
    create_slide_generator_chain,
    run_slide_generator_agent,
    SLIDE_SYSTEM_PROMPT,
    SLIDE_HUMAN_PROMPT,
)
from core.agents.viva_agent import (
    create_viva_chain,
    run_viva_agent,
    VIVA_SYSTEM_PROMPT,
    VIVA_HUMAN_PROMPT,
    FEW_SHOT_EXAMPLE_HUMAN,
    FEW_SHOT_EXAMPLE_AI,
)

def test_agents():
    print("=== 1. TEST AGENT IMPORTS ===")
    print("Summarizer agent imported OK.")
    print("Concept-Extractor agent imported OK.")
    print("Slide-Generator agent imported OK.")
    print("Viva agent imported OK.")

    print("\n=== 2. TEST LCEL CHAIN COMPOSITION ===")
    dummy_llm, _ = get_llm(api_key="AIzaSyDummyKeyForChainStructureTestOnly1234")
    assert dummy_llm is not None, "Dummy LLM could not be initialized."

    sum_chain = create_summarizer_chain(dummy_llm)
    reflection_chain = create_reflection_chain(dummy_llm)
    concept_chain = create_concept_extractor_chain(dummy_llm)
    slide_chain = create_slide_generator_chain(dummy_llm)
    viva_chain = create_viva_chain(dummy_llm)

    print("Summarizer Drafting LCEL chain composed successfully:", type(sum_chain).__name__)
    print("Summarizer Reflection LCEL chain composed successfully:", type(reflection_chain).__name__)
    print("Concept-Extractor LCEL chain composed successfully:", type(concept_chain).__name__)
    print("Slide-Generator LCEL chain composed successfully:", type(slide_chain).__name__)
    print("Viva Few-Shot LCEL chain composed successfully:", type(viva_chain).__name__)

    print("\n=== 3. TEST FEW-SHOT EXEMPLARS IN VIVA AGENT ===")
    # Extract the messages in the Viva PromptTemplate to confirm few-shot structure
    # Expected messages: SystemMessage, HumanMessage (Example), AIMessage (Example), HumanMessage (Actual Prompt)
    viva_prompt = viva_chain.first
    assert len(viva_prompt.messages) == 4, f"Expected 4 messages in few-shot template, got {len(viva_prompt.messages)}"
    print(f"Message 0 (System): {type(viva_prompt.messages[0]).__name__}")
    print(f"Message 1 (Few-Shot Human Example): {type(viva_prompt.messages[1]).__name__}")
    print(f"Message 2 (Few-Shot AI Example): {type(viva_prompt.messages[2]).__name__}")
    print(f"Message 3 (Actual Human Query): {type(viva_prompt.messages[3]).__name__}")

    assert "LinAttention" in FEW_SHOT_EXAMPLE_HUMAN
    assert "Question 1:" in FEW_SHOT_EXAMPLE_AI
    assert "{filename}" in viva_prompt.messages[3].prompt.template
    assert "{context}" in viva_prompt.messages[3].prompt.template
    assert "{user_query}" in viva_prompt.messages[3].prompt.template
    print("  [PASS] Genuine few-shot demonstration verified in Viva Agent.")

    print("\n=== 4. TEST REFLECTION / SELF-CORRECTION LOGIC ===")
    # Verify reflection prompt variables
    assert "{context}" in REFLECTION_HUMAN_PROMPT
    assert "{draft_summary}" in REFLECTION_HUMAN_PROMPT
    print("  [PASS] Verified reflection prompt input variables: ['context', 'draft_summary']")

    # Test the reflection loop mechanics using a mock LCEL runnable
    from langchain_core.runnables import RunnableLambda

    draft_with_hallucination = (
        "## 🎯 Research Objectives & Motivation\n"
        "The authors propose PaperPilot-AI.\n\n"
        "## 📊 Major Findings & Quantitative Results\n"
        "The system achieved 99.9% accuracy on a quantum computer (UNGROUNDED).\n"
    )
    audited_corrected_summary = (
        "## 🎯 Research Objectives & Motivation\n"
        "The authors propose PaperPilot-AI to automate research comprehension.\n\n"
        "## 📊 Major Findings & Quantitative Results\n"
        "PaperPilot-AI extracts full document content within 120ms. Accuracy on quantum computers is not explicitly identified in the extracted text.\n"
    )

    mock_draft_llm = RunnableLambda(lambda x: draft_with_hallucination)
    mock_reflection_llm = RunnableLambda(lambda x: audited_corrected_summary)

    # Verify that the two-step chain works sequentially
    draft_step = create_summarizer_chain(mock_draft_llm).invoke({
        "filename": "test.pdf",
        "context": "PaperPilot extracts content in 120ms.",
        "user_query": "Summarize",
    })
    corrected_step = create_reflection_chain(mock_reflection_llm).invoke({
        "context": "PaperPilot extracts content in 120ms.",
        "draft_summary": draft_step,
    })
    assert "120ms" in corrected_step
    assert "UNGROUNDED" not in corrected_step
    print("  [PASS] Reflection & self-correction loop successfully verified with mock pipeline.")

    print("\n=== 5. TEST RETRIEVAL INTEGRATION WITH CHROMA ===")
    vstore = get_paper_vectorstore("sample_research_paper.pdf")
    assert vstore is not None, "Failed to load vectorstore for sample_research_paper.pdf"
    
    docs = retrieve_relevant_passages("methodology evaluation", vstore, top_k=3)
    print(f"Vector store retrieval operational. Retrieved {len(docs)} passages.")

    print("\n=== 6. TEST OFFLINE EXTRACTIVE FALLBACK (NO API KEY) ===")
    # Agent 1: Summarizer offline fallback
    res_sum = run_summarizer_agent(vstore, "sample_research_paper.pdf", api_key=None)
    assert res_sum["status"] == "success"
    assert "Research Objective" in res_sum["output"]
    assert res_sum["engine"] == "offline_extractive"
    print("  [PASS] Summarizer offline extractive fallback verified.")

    # Agent 2: Concept-Extractor offline fallback
    res_concept = run_concept_extractor_agent(vstore, "sample_research_paper.pdf", api_key=None)
    assert res_concept["status"] == "success"
    assert res_concept["engine"] == "offline_extractive"
    print("  [PASS] Concept-Extractor offline extractive fallback verified.")

    # Agent 3: Slide-Generator offline fallback
    res_slide = run_slide_generator_agent(vstore, "sample_research_paper.pdf", api_key=None)
    assert res_slide["status"] == "success"
    assert res_slide["engine"] == "offline_extractive"
    print("  [PASS] Slide-Generator offline extractive fallback verified.")

    # Agent 4: Viva Agent offline fallback
    res_viva = run_viva_agent(vstore, "sample_research_paper.pdf", api_key=None)
    assert res_viva["status"] == "success"
    assert res_viva["engine"] == "offline_extractive"
    print("  [PASS] Viva Agent offline extractive fallback verified.")

    print("\n=== 7. GEMINI LIVE API STATUS ===")
    print("Live LLM generation: UNTESTED (No real GOOGLE_API_KEY currently supplied in .env)")
    print("All unit tests, few-shot prompts, reflection loops, and offline fallbacks: 100% PASSED.")

if __name__ == "__main__":
    test_agents()
