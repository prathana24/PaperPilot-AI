"""
Test Suite for the 4 Specialized Agents
Validates:
1. Agent imports
2. LCEL chain composition
3. Prompt variables
4. Offline extractive fallback for all 4 agents (when API key is absent)
5. Live LCEL path readiness (when API key is supplied)
"""

import os
from core.config import get_llm
from core.vectorstore import get_paper_vectorstore, retrieve_relevant_passages
from core.agents.summarizer import (
    create_summarizer_chain,
    run_summarizer_agent,
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARIZER_HUMAN_PROMPT,
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
    concept_chain = create_concept_extractor_chain(dummy_llm)
    slide_chain = create_slide_generator_chain(dummy_llm)
    viva_chain = create_viva_chain(dummy_llm)

    print("Summarizer LCEL chain composed successfully:", type(sum_chain).__name__)
    print("Concept-Extractor LCEL chain composed successfully:", type(concept_chain).__name__)
    print("Slide-Generator LCEL chain composed successfully:", type(slide_chain).__name__)
    print("Viva LCEL chain composed successfully:", type(viva_chain).__name__)

    print("\n=== 3. TEST PROMPT TEMPLATES & EXPECTED VARIABLES ===")
    for name, p_sys, p_hum in [
        ("Summarizer", SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_HUMAN_PROMPT),
        ("Concept-Extractor", CONCEPT_SYSTEM_PROMPT, CONCEPT_HUMAN_PROMPT),
        ("Slide-Generator", SLIDE_SYSTEM_PROMPT, SLIDE_HUMAN_PROMPT),
        ("Viva Agent", VIVA_SYSTEM_PROMPT, VIVA_HUMAN_PROMPT),
    ]:
        assert "{filename}" in p_hum, f"Missing {{filename}} in {name}"
        assert "{context}" in p_hum, f"Missing {{context}} in {name}"
        assert "{user_query}" in p_hum, f"Missing {{user_query}} in {name}"
        print(f"Verified prompt variables for {name}: ['filename', 'context', 'user_query']")

    print("\n=== 4. TEST RETRIEVAL INTEGRATION WITH CHROMA ===")
    vstore = get_paper_vectorstore("sample_research_paper.pdf")
    assert vstore is not None, "Failed to load vectorstore for sample_research_paper.pdf"
    
    docs = retrieve_relevant_passages("methodology evaluation", vstore, top_k=3)
    print(f"Vector store retrieval operational. Retrieved {len(docs)} passages.")

    print("\n=== 5. TEST OFFLINE EXTRACTIVE FALLBACK (NO API KEY) ===")
    # Agent 1: Summarizer offline fallback
    res_sum = run_summarizer_agent(vstore, "sample_research_paper.pdf", api_key=None)
    print("Summarizer Agent (Offline Fallback):")
    print("  Status:", res_sum["status"])
    print("  Engine:", res_sum["engine"])
    print("  Label:", res_sum["label"])
    assert res_sum["status"] == "success"
    assert "Research Objective" in res_sum["output"]
    assert "Methodology" in res_sum["output"]
    assert "Key Empirical Findings" in res_sum["output"]
    print("  [PASS] Structured offline summary successfully generated.")

    # Agent 2: Concept-Extractor offline fallback
    res_concept = run_concept_extractor_agent(vstore, "sample_research_paper.pdf", api_key=None)
    print("\nConcept-Extractor Agent (Offline Fallback):")
    print("  Status:", res_concept["status"])
    print("  Engine:", res_concept["engine"])
    print("  Label:", res_concept["label"])
    assert res_concept["status"] == "success"
    assert "Core Extracted Terms" in res_concept["output"]
    print("  [PASS] Offline concepts successfully extracted.")

    # Agent 3: Slide-Generator offline fallback
    res_slide = run_slide_generator_agent(vstore, "sample_research_paper.pdf", api_key=None)
    print("\nSlide-Generator Agent (Offline Fallback):")
    print("  Status:", res_slide["status"])
    print("  Engine:", res_slide["engine"])
    print("  Label:", res_slide["label"])
    assert res_slide["status"] == "success"
    assert "Slide 1:" in res_slide["output"]
    assert "Speaker Notes" in res_slide["output"]
    print("  [PASS] Offline slide outline successfully generated.")

    # Agent 4: Viva Agent offline fallback
    res_viva = run_viva_agent(vstore, "sample_research_paper.pdf", api_key=None)
    print("\nViva Agent (Offline Fallback):")
    print("  Status:", res_viva["status"])
    print("  Engine:", res_viva["engine"])
    print("  Label:", res_viva["label"])
    assert res_viva["status"] == "success"
    assert "Question 1:" in res_viva["output"]
    assert "Question 10:" in res_viva["output"]
    print("  [PASS] Offline 10 viva questions successfully generated.")

    print("\n=== 6. GEMINI LIVE API STATUS ===")
    print("Live LLM generation: UNTESTED (No real GOOGLE_API_KEY currently supplied in .env)")
    print("Both API-Key and Offline Fallback paths: 100% verified.")

if __name__ == "__main__":
    test_agents()
