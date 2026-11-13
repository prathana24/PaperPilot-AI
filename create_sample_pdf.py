import pymupdf

def create_sample_paper(filepath="sample_research_paper.pdf"):
    doc = pymupdf.open()
    page = doc.new_page()

    sample_text = """PaperPilot-AI: An Intelligent Research Assistant for Scientific Papers

Abstract
Understanding modern machine learning and computer science research papers presents significant hurdles for novice researchers and undergraduate students. In this paper, we propose PaperPilot-AI, a lightweight, accessible framework designed to parse, analyze, and synthesize academic publications. We evaluate our approach across various document styles and demonstrate substantial improvements in reading comprehension efficiency.

1. Introduction and Objectives
The main objective of this study is to demystify complex research literature by automating document parsing, key section identification, and semantic question-answering. Contemporary academic literature is growing at an unprecedented pace, leaving students overwhelmed with technical jargon and dense mathematics. Our goal is to provide an intuitive interface that summarizes research contributions without requiring prohibitive compute infrastructure.

2. Methodology and Framework
Our methodology combines fast document extraction with PyMuPDF and frequency-weighted extractive sentence ranking. First, the PDF byte stream is decompressed and parsed page by page to recover sequential textual hierarchy. Second, an adaptive stopword-filtered term frequency scoring model evaluates sentence centrality. Third, section-boundary regex heuristics categorize key findings into research objectives, methodology, experimental results, and concluding remarks. Finally, a keyword-relevance matching algorithm enables interactive passage retrieval.

3. Results and Performance Evaluation
Experimental evaluation shows that PaperPilot-AI extracts full document content within 120 milliseconds on commodity hardware. Comprehension benchmarks indicate that students utilizing PaperPilot-AI identify the central research objective 68% faster compared to traditional manual reading. The TF-IDF based sentence extractor achieves high alignment with author-provided abstracts while introducing zero dependency bloat.

4. Conclusion and Future Directions
In conclusion, PaperPilot-AI offers an efficient, beginner-friendly pipeline to unpack technical papers. We have demonstrated that accessible semantic parsing bridges the gap between complex research and student comprehension. Future work will investigate real-time citation network graph visualization and multilingual document parsing.
"""

    # Insert text into page
    rect = pymupdf.Rect(50, 50, 550, 750)
    page.insert_textbox(rect, sample_text, fontsize=11, fontname="helv", align=0)
    doc.save(filepath)
    doc.close()
    print(f"Sample PDF created successfully at: {filepath}")

if __name__ == "__main__":
    create_sample_paper()
