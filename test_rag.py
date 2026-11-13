import os
import pymupdf
import core.vectorstore as vs

def run_tests():
    print("=== 1. TEST TEXT EXTRACTION & CHUNKING ===")
    doc = pymupdf.open("sample_research_paper.pdf")
    pages_dict = {}
    for i, page in enumerate(doc):
        pages_dict[i + 1] = page.get_text()
    doc.close()

    chunks = vs.chunk_document_pages(pages_dict, "sample_research_paper.pdf", chunk_size=1000, chunk_overlap=150)
    print(f"Total chunks created: {len(chunks)}")
    for idx, c in enumerate(chunks):
        print(f"Chunk {idx}: Page {c.metadata['page']}, Length {len(c.page_content)} chars")
        print(f"  Preview: {c.page_content[:90]}...")

    print("\n=== 2. TEST CHROMA VECTORSTORE INDEXING ===")
    vstore = vs.index_paper_chunks(chunks, "sample_research_paper.pdf")
    print("Chroma collection created successfully.")
    print(f"Total items in collection: {vstore._collection.count()}")

    print("\n=== 3. TEST SEMANTIC SIMILARITY RETRIEVAL ===")
    query = "What is the main objective of PaperPilot-AI?"
    retrieved = vs.retrieve_relevant_passages(query, vstore, top_k=2)
    print(f'Query: "{query}"')
    print(f"Retrieved results count: {len(retrieved)}")
    for i, r in enumerate(retrieved):
        print(f"Result {i+1} (Page {r['page']}, Distance Score: {r['score']:.4f}):")
        print(f"  Content: {r['content'][:140]}...")

    print("\n=== 4. TEST PERSISTENCE ACROSS RE-INITIALIZATION ===")
    reloaded_vstore = vs.get_paper_vectorstore("sample_research_paper.pdf")
    print("Persistent vectorstore reloaded successfully:", reloaded_vstore is not None)
    if reloaded_vstore:
        print(f"Persistent collection count: {reloaded_vstore._collection.count()}")

    print("\n=== 5. TEST EDGE CASES (EMPTY QUERY, EMPTY DOC) ===")
    empty_res = vs.retrieve_relevant_passages("", vstore)
    print("Empty query returned items:", len(empty_res))
    empty_chunks = vs.chunk_document_pages({}, "dummy.pdf")
    print("Empty pages returned chunks:", len(empty_chunks))
    
    print("\n=== 6. VERIFY PERSISTENT FILES ON DISK ===")
    if os.path.exists("chroma_db"):
        print("chroma_db directory exists:", os.path.exists("chroma_db"))
        contents = os.listdir("chroma_db")
        print("chroma_db files:", contents)

if __name__ == "__main__":
    run_tests()
