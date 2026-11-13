"""
Test Suite for Conversation Memory (core/memory.py)
Validates turn recording, history retrieval, history trimming,
memory clearing, document isolation, and session isolation.
"""

from core.memory import ConversationMemory, SessionMemoryManager

def run_memory_tests():
    print("=== 1. TEST CONVERSATION STORAGE & RETRIEVAL ===")
    mem = ConversationMemory(document_id="paper_a.pdf", max_turns=4)
    mem.add_user_message("What is PaperPilot-AI?")
    mem.add_assistant_message("It is an academic research assistant.", intent="ask_question", agent="Q&A Engine")

    history = mem.get_recent_history()
    assert len(history) == 2, f"Expected 2 turns, got {len(history)}"
    assert history[0].role == "user"
    assert history[0].content == "What is PaperPilot-AI?"
    assert history[1].role == "assistant"
    assert history[1].agent == "Q&A Engine"
    print("  [PASS] Stored and retrieved 2 conversational turns accurately.")

    print("\n=== 2. TEST MAX TURNS TRIMMING & FORMATTING ===")
    # Add 4 more turns (total 6, exceeding max_turns=4)
    mem.add_user_message("Q2")
    mem.add_assistant_message("A2")
    mem.add_user_message("Q3")
    mem.add_assistant_message("A3")

    trimmed_history = mem.get_recent_history()
    assert len(trimmed_history) == 4, f"Expected 4 turns after trimming, got {len(trimmed_history)}"
    assert trimmed_history[0].content == "Q2", f"Oldest remaining turn should be Q2, got {trimmed_history[0].content}"
    assert trimmed_history[-1].content == "A3", f"Newest turn should be A3, got {trimmed_history[-1].content}"
    
    formatted = mem.get_formatted_history(limit=2)
    assert "User: Q3" in formatted and "Assistant: A3" in formatted
    print("  [PASS] Trimming correctly capped history to max_turns=4.")
    print("  [PASS] Formatted history for LLM prompt:\n" + formatted)

    print("\n=== 3. TEST MEMORY CLEARING ===")
    mem.clear()
    assert len(mem.get_recent_history()) == 0, "History should be empty after clear()"
    print("  [PASS] Memory clear verified.")

    print("\n=== 4. TEST DOCUMENT ISOLATION ===")
    manager = SessionMemoryManager()
    session_id = "session_user_1"

    # User uploads Paper 1
    mem_doc1 = manager.get_memory(session_id, "quantum_computing.pdf")
    mem_doc1.add_user_message("Explain qubits")
    mem_doc1.add_assistant_message("A qubit is the basic unit of quantum information.")

    # User uploads Paper 2 in the same session
    mem_doc2 = manager.get_memory(session_id, "deep_learning.pdf")
    mem_doc2.add_user_message("Explain backpropagation")
    mem_doc2.add_assistant_message("Backpropagation computes gradients.")

    # Verify Paper 1 does not contain Paper 2 data
    doc1_history = mem_doc1.get_recent_history()
    doc2_history = mem_doc2.get_recent_history()

    assert len(doc1_history) == 2
    assert "qubits" in doc1_history[0].content
    assert "backpropagation" not in [t.content for t in doc1_history]

    assert len(doc2_history) == 2
    assert "backpropagation" in doc2_history[0].content
    assert "qubits" not in [t.content for t in doc2_history]
    print("  [PASS] Document isolation confirmed: No cross-paper pollution.")

    print("\n=== 5. TEST SESSION ISOLATION ===")
    session_a = "user_alpha"
    session_b = "user_beta"

    mem_a = manager.get_memory(session_a, "shared_paper.pdf")
    mem_b = manager.get_memory(session_b, "shared_paper.pdf")

    mem_a.add_user_message("Alpha private note")
    mem_b.add_user_message("Beta private note")

    assert "Alpha private note" in [t.content for t in mem_a.get_recent_history()]
    assert "Alpha private note" not in [t.content for t in mem_b.get_recent_history()]
    assert "Beta private note" in [t.content for t in mem_b.get_recent_history()]
    assert "Beta private note" not in [t.content for t in mem_a.get_recent_history()]
    print("  [PASS] Session isolation confirmed: Separate users/sessions cannot view each other's memories.")

    print("\n=== MEMORY TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_memory_tests()
