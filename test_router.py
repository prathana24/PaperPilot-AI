"""
Test Suite for Intent Detection and Agent Routing (core/router.py)
Validates rule-based classification, ambiguous input detection,
agent routing, and grounded QA fallback.
"""

from core.router import (
    classify_intent_rule_based,
    classify_intent,
    route_and_execute,
    INTENT_SUMMARIZE,
    INTENT_CONCEPTS,
    INTENT_SLIDES,
    INTENT_VIVA,
    INTENT_QA,
    INTENT_AMBIGUOUS,
    INTENT_MAP,
)
from core.vectorstore import get_paper_vectorstore
from core.memory import ConversationMemory

def run_router_tests():
    print("=== 1. TEST ALL 5 INTENT CATEGORIES (RULE-BASED) ===")
    test_cases = [
        ("Can you summarize this paper for me?", INTENT_SUMMARIZE),
        ("Give me an executive summary and overview", INTENT_SUMMARIZE),
        ("What algorithms and theoretical concepts are used?", INTENT_CONCEPTS),
        ("Explain the methodology and architecture", INTENT_CONCEPTS),
        ("Generate 8 presentation slides for my talk", INTENT_SLIDES),
        ("Make a PowerPoint deck outline with speaker notes", INTENT_SLIDES),
        ("Prepare 10 viva exam questions with answers", INTENT_VIVA),
        ("Give me oral defense questions to practice", INTENT_VIVA),
        ("What dataset was used for evaluation?", INTENT_QA),
        ("How fast does the parsing execute?", INTENT_QA),
    ]

    for query, expected_intent in test_cases:
        detected = classify_intent_rule_based(query)
        assert detected == expected_intent, f"Failed for '{query}': expected {expected_intent}, got {detected}"
        print(f"  [PASS] '{query[:35]}...' -> {detected} ({INTENT_MAP[detected]})")

    print("\n=== 2. TEST AMBIGUOUS INTENT HANDLING ===")
    ambiguous_queries = [
        "",
        "   ",
        "hello",
        "summarize this paper and generate viva questions for slides", # multiple conflicts
    ]
    for amb in ambiguous_queries:
        detected = classify_intent_rule_based(amb)
        assert detected == INTENT_AMBIGUOUS, f"Expected ambiguous for '{amb}', got {detected}"
        print(f"  [PASS] '{amb}' -> {detected} ({INTENT_MAP[detected]})")

    print("\n=== 3. TEST AGENT ROUTING & STRUCTURED PAYLOAD ===")
    vstore = get_paper_vectorstore("sample_research_paper.pdf")
    assert vstore is not None, "Failed to load sample vectorstore"
    mem = ConversationMemory(document_id="sample_research_paper.pdf")

    # Test ambiguous routing handling
    res_amb = route_and_execute("hello", vstore, "sample_research_paper.pdf", memory=mem, api_key=None)
    print("Ambiguous route result:")
    print("  Status:", res_amb["status"])
    print("  Intent:", res_amb["intent"])
    print("  Selected Agent:", res_amb["agent"])
    print("  Disambiguation prompt provided:", "Would you like me to:" in res_amb["output"])

    # Test Grounded Q&A routing without API key (fallback retrieval)
    res_qa = route_and_execute(
        "What is the main objective of this study?",
        vstore,
        "sample_research_paper.pdf",
        memory=mem,
        api_key=None,
    )
    print("\nQ&A routing result (without API key):")
    print("  Status:", res_qa["status"])
    print("  Intent:", res_qa["intent"])
    print("  Selected Agent:", res_qa["agent"])
    print("  Engine:", res_qa["engine"])
    print("  Source passages returned:", len(res_qa["sources"]))
    assert len(res_qa["sources"]) > 0, "No sources returned in Q&A"
    print("  Output preview:", res_qa["output"][:140], "...")

    # Test Explicit Intent Override
    res_explicit = route_and_execute(
        "Give me the deck",
        vstore,
        "sample_research_paper.pdf",
        memory=mem,
        api_key=None,
        explicit_intent=INTENT_SLIDES,
    )
    print("\nExplicit Intent Override result:")
    print("  Detected Intent:", res_explicit["intent"])
    print("  Selected Agent:", res_explicit["agent"])
    print("  Classifier Engine:", res_explicit["classifier_engine"])

    print("\n=== ROUTER TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_router_tests()
