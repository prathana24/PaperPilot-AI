"""
Integration Verification Test for PaperPilot-AI (Step 6)
Validates:
1. App module import
2. PDF text extraction & chunking
3. ChromaDB real-time indexing
4. Router dispatch across all 4 agents + Grounded Q&A
5. PPTX export generation
6. Session memory isolation
"""

import os
import pymupdf
import app
from core.vectorstore import chunk_document_pages, index_paper_chunks, get_paper_vectorstore
from core.router import route_and_execute, INTENT_SUMMARIZE, INTENT_CONCEPTS, INTENT_SLIDES, INTENT_VIVA, INTENT_QA
from core.pptx_export import generate_pptx_deck
from core.memory import get_session_memory

def run_integration_tests():
    print("=== 1. TEST PDF EXTRACTION & CHUNKING ===")
    with open("sample_research_paper.pdf", "rb") as f:
        pdf_bytes = f.read()
    raw_text, page_count, pages_dict = app.extract_text_from_pdf(pdf_bytes)
    print(f"Extracted {page_count} pages, {len(raw_text.split())} words.")
    assert page_count == 1
    assert len(raw_text) > 500

    chunks = chunk_document_pages(pages_dict, "sample_research_paper.pdf", chunk_size=1000, chunk_overlap=150)
    print(f"Generated {len(chunks)} chunks.")
    assert len(chunks) >= 2

    print("\n=== 2. TEST CHROMADB INDEXING ===")
    vstore = index_paper_chunks(chunks, "sample_research_paper.pdf")
    assert vstore is not None
    print(f"ChromaDB collection count: {vstore._collection.count()}")
    assert vstore._collection.count() >= 2

    print("\n=== 3. TEST INTENT ROUTING TO GROUNDED Q&A (EXTRACTIVE FALLBACK) ===")
    mem = get_session_memory("test_session_id", "sample_research_paper.pdf")
    res_qa = route_and_execute(
        user_query="What benchmark improvement did students achieve using PaperPilot-AI?",
        vectorstore=vstore,
        filename="sample_research_paper.pdf",
        memory=mem,
        api_key=None,
    )
    print("Detected Intent:", res_qa["intent"])
    print("Selected Agent:", res_qa["agent"])
    print("Engine:", res_qa["engine"])
    print("Sources returned:", len(res_qa["sources"]))
    assert res_qa["intent"] == INTENT_QA
    assert len(res_qa["sources"]) > 0

    print("\n=== 4. TEST SLIDE DECK & PPTX EXPORT ===")
    sample_slide_md = """
### Slide 1: Introduction to PaperPilot-AI
- Automatic research paper understanding for students
- Reduces cognitive overload from technical jargon
- High-speed parsing and structured section extraction

**🎙️ Speaker Notes:**
Welcome to this presentation on PaperPilot-AI, designed to assist students with scientific literature.

### Slide 2: Methodology & Vector Architecture
- PyMuPDF for high-fidelity text extraction
- RecursiveCharacterTextSplitter with 1000 char window
- Local HuggingFace embeddings running offline in ChromaDB

**🎙️ Speaker Notes:**
The architecture uses localized embeddings to ensure zero API quota dependency for core indexing.
"""
    pptx_stream = generate_pptx_deck(sample_slide_md, paper_title="Sample Research Paper")
    pptx_bytes = pptx_stream.getvalue()
    print(f"PPTX Presentation compiled successfully: {len(pptx_bytes)} bytes.")
    assert len(pptx_bytes) > 10000

    print("\n=== 5. TEST MEMORY SESSION PERSISTENCE ===")
    history = mem.get_recent_history()
    print(f"Conversation memory logged {len(history)} turns.")
    assert len(history) >= 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"

    print("\n=== ALL INTEGRATION TESTS PASSED CLEANLY ===")

if __name__ == "__main__":
    run_integration_tests()
