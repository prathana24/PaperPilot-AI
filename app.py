"""
PaperPilot-AI: An AI-Powered Research Paper Assistant
Full Agentic RAG implementation featuring:
- PyMuPDF text extraction
- RecursiveCharacterTextSplitter & local HuggingFace embeddings
- Persistent ChromaDB vector database
- 4 Specialized LCEL Agents (Summarizer, Concept-Extractor, Slide-Generator, Viva Agent)
- Intent Detection & Routing Engine
- Grounded Q&A with session-scoped memory
- Presentation Export (.pptx & Markdown)
- Robust Offline Fallback when Gemini API is unconfigured
- White & light-blue modern UI
"""

import os
import io
import re
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pymupdf  # Modern PyMuPDF API

# Core imports
from core.config import get_google_api_key, get_llm, get_gemini_model_name
from core.vectorstore import (
    chunk_document_pages,
    index_paper_chunks,
    get_paper_vectorstore,
    retrieve_relevant_passages,
)
from core.memory import get_session_memory
from core.router import (
    route_and_execute,
    INTENT_SUMMARIZE,
    INTENT_CONCEPTS,
    INTENT_SLIDES,
    INTENT_VIVA,
    INTENT_QA,
)
from core.pptx_export import generate_pptx_deck, PPTX_AVAILABLE

# -----------------------------------------------------------------------------
# Configuration and Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PaperPilot-AI | Agentic Research Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* White and Light-Blue Institutional Theme */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    .main-header {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #075985 100%);
        padding: 22px 28px;
        border-radius: 14px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.15);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #ffffff !important;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 1.0rem;
        color: #e0f2fe;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #bae6fd;
        border-left: 5px solid #0284c7;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .section-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
    }
    .badge-amber {
        background-color: #fef3c7;
        color: #b45309;
    }
    .qa-box {
        background-color: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .source-box {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 8px 14px;
        margin: 6px 0;
        font-size: 0.88rem;
        color: #334155;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PDF Processing (PyMuPDF)
# -----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, int, Dict[int, str]]:
    """Extracts clean text from PDF bytes using PyMuPDF."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages_dict = {}
    full_text_list = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        cleaned_page_text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages_dict[page_num + 1] = cleaned_page_text
        if cleaned_page_text:
            full_text_list.append(cleaned_page_text)

    doc.close()
    full_text = "\n\n".join(full_text_list)
    return full_text, len(pages_dict), pages_dict


# -----------------------------------------------------------------------------
# Application State Initialization
# -----------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_default_session"

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = {
        "filename": None,
        "text": "",
        "page_count": 0,
        "word_count": 0,
        "pages_dict": {},
        "indexed": False,
        "chunks_count": 0,
    }

if "agent_outputs" not in st.session_state:
    st.session_state.agent_outputs = {
        "summary": None,
        "concepts": None,
        "slides": None,
        "viva": None,
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------------------------------------------------------
# Sidebar Configuration & Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ✈️ **PaperPilot-AI**")
    st.caption("Agentic Research Assistant with ChromaDB & LCEL")
    st.markdown("---")

    # API Configuration
    st.subheader("🔑 Gemini API Setup")
    env_key = get_google_api_key()
    user_api_key = st.text_input(
        "Enter Google API Key (optional):",
        value=env_key or "",
        type="password",
        help="Reads from .env automatically if set. Required for Generative AI agents.",
    )
    effective_api_key = user_api_key if user_api_key.strip() else env_key
    model_name = get_gemini_model_name()

    if effective_api_key:
        st.success(f"API Key detected • Model: {model_name}")
    else:
        st.info("No API Key configured. Offline extractive engines & local ChromaDB are active.")

    st.markdown("---")

    # Document Upload
    st.subheader("📑 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a Research Paper (PDF)",
        type=["pdf"],
        help="Upload scientific publications, preprints, or technical reports."
    )

    if uploaded_file is not None and uploaded_file.name != st.session_state.pdf_data.get("filename"):
        with st.spinner("Parsing PDF with PyMuPDF and indexing into ChromaDB..."):
            file_bytes = uploaded_file.read()
            raw_text, pages, pages_dict = extract_text_from_pdf(file_bytes)
            words = len(raw_text.split())

            if not raw_text.strip():
                st.error("Uploaded PDF does not contain extractable text (it may be a scanned image).")
            else:
                chunks = chunk_document_pages(pages_dict, uploaded_file.name, chunk_size=1000, chunk_overlap=150)
                index_paper_chunks(chunks, uploaded_file.name)

                st.session_state.pdf_data = {
                    "filename": uploaded_file.name,
                    "text": raw_text,
                    "page_count": pages,
                    "word_count": words,
                    "pages_dict": pages_dict,
                    "indexed": True,
                    "chunks_count": len(chunks),
                }
                # Reset previous outputs for new paper
                st.session_state.agent_outputs = {
                    "summary": None,
                    "concepts": None,
                    "slides": None,
                    "viva": None,
                }
                doc_memory = get_session_memory(st.session_state.session_id, uploaded_file.name)
                st.session_state.chat_history = [t.to_dict() for t in doc_memory.get_recent_history()]
                st.success(f"Indexed {len(chunks)} chunks into ChromaDB!")

    st.markdown("---")
    st.subheader("📌 Navigation")
    menu_choice = st.radio(
        "Select View:",
        [
            "Overview & Metrics",
            "Research Paper Summary",
            "Concept Extraction",
            "Presentation Generator",
            "Viva Preparation",
            "Ask Your Paper (Interactive Chat)",
            "Extracted Raw Text",
        ],
    )

    st.markdown("---")
    st.caption("PaperPilot-AI v2.0 • LangChain • ChromaDB • Python 3.10")


# -----------------------------------------------------------------------------
# Main Header Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>✈️ PaperPilot-AI</h1>
        <p>Demystifying complex research papers using Agentic RAG, LangChain LCEL, and local ChromaDB vector retrieval.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

filename = st.session_state.pdf_data["filename"]
full_text = st.session_state.pdf_data["text"]
vectorstore = get_paper_vectorstore(filename) if filename else None

if filename:
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge badge-success">Active Document</span><br>
                <strong>{filename}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s2:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge">Document Size</span><br>
                <strong>{st.session_state.pdf_data['page_count']} Pages • {st.session_state.pdf_data['word_count']:,} Words</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s3:
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge badge-success">ChromaDB Index</span><br>
                <strong>{st.session_state.pdf_data['chunks_count']} Chunks Indexed</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_s4:
        engine_label = f"Gemini ({model_name})" if effective_api_key else "Offline Extractive RAG"
        st.markdown(
            f"""
            <div class="metric-card">
                <span class="badge {'badge-success' if effective_api_key else 'badge-amber'}">AI Mode</span><br>
                <strong>{engine_label}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("👆 Please upload a research paper (PDF) using the sidebar to begin analysis.")


def render_source_citations(sources: List[Dict[str, Any]]):
    """Renders structured source citations in an expander."""
    if not sources:
        return
    with st.expander(f"📚 Grounded Citations & Sources ({len(sources)} Passages Retrieved)", expanded=False):
        for idx, src in enumerate(sources):
            score_txt = f"Distance: {src['score']:.4f}" if "score" in src else ""
            st.markdown(
                f"""
                <div class="source-box">
                    <strong>Citation [{idx + 1}]</strong> — Page {src.get('page', 1)} | {score_txt}<br>
                    <em>Document: {src.get('source', 'Unknown')}</em>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------------
# VIEW 1: Overview & Metrics
# -----------------------------------------------------------------------------
if menu_choice == "Overview & Metrics":
    st.markdown("### 📊 Document Overview & RAG Architecture")

    if not filename:
        st.warning("No PDF uploaded yet. Use the sidebar to upload a research paper.")
    else:
        st.markdown(
            f"""
            <div class="section-card">
                <h4 style="color: #0284c7; margin-top: 0;">📄 Document Specification</h4>
                <ul>
                    <li><strong>File Name:</strong> {filename}</li>
                    <li><strong>Total Pages:</strong> {st.session_state.pdf_data['page_count']}</li>
                    <li><strong>Word Count:</strong> {st.session_state.pdf_data['word_count']:,}</li>
                    <li><strong>Vector Store Status:</strong> Persistent ChromaDB collection indexed with 384-dim HuggingFace embeddings</li>
                    <li><strong>Total Text Chunks:</strong> {st.session_state.pdf_data['chunks_count']} (Recursive splitter: 1000 char size, 150 overlap)</li>
                    <li><strong>Current Operational Mode:</strong> {'Gemini LLM Active' if effective_api_key else 'Offline Extractive Mode (Fully Demonstrable without API Key)'}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <h4 style="color: #0284c7; margin-top: 0;">🤖 The 4 Specialized AI Agents</h4>
                <ol>
                    <li><strong>Summarizer Agent:</strong> Generates multi-perspective academic summaries grounded in retrieved ChromaDB context.</li>
                    <li><strong>Concept-Extractor Agent:</strong> Unpacks core theoretical principles, algorithms, and empirical benchmarks into student-friendly terms.</li>
                    <li><strong>Slide-Generator Agent:</strong> Transforms research discoveries into an 8-slide conference deck with speaker notes and downloadable .pptx export.</li>
                    <li><strong>Viva Agent:</strong> Formulates 10 targeted oral defense examination questions with model answers and examiner focus tips.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# VIEW 2: Research Paper Summary (Agent 1)
# -----------------------------------------------------------------------------
elif menu_choice == "Research Paper Summary":
    st.markdown("### 📝 Agent 1: Structured Research Paper Summary")

    if not filename:
        st.warning("Please upload a paper from the sidebar.")
    else:
        if not effective_api_key:
            st.info("ℹ️ **Offline Mode Active:** Generating structured extractive summary directly from paper text without API keys.")

        button_label = "✨ Generate Structured Summary" if effective_api_key else "⚡ Generate Offline Extractive Summary"
        if st.button(button_label, key="btn_run_summary"):
            with st.spinner("Summarizer Agent is analyzing paper text..."):
                res = route_and_execute(
                    user_query="Generate a comprehensive structured academic summary of this paper.",
                    vectorstore=vectorstore,
                    filename=filename,
                    api_key=effective_api_key,
                    full_text=full_text,
                    explicit_intent=INTENT_SUMMARIZE,
                )
                st.session_state.agent_outputs["summary"] = res

        summary_res = st.session_state.agent_outputs.get("summary")
        if summary_res and summary_res.get("status") == "success":
            badge_class = "badge-success" if "gemini" in summary_res.get("engine", "") else "badge-amber"
            st.markdown(
                f"""
                <div class="section-card">
                    <span class="badge {badge_class}">{summary_res.get('label', 'Summary')}</span>
                    <div style="margin-top: 14px; font-size: 1.05rem; line-height: 1.7;">
                        {summary_res['output']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_source_citations(summary_res.get("sources", []))

            st.download_button(
                label="📥 Download Summary as Text File",
                data=summary_res["output"],
                file_name=f"{os.path.splitext(filename)[0]}_Summary.txt",
                mime="text/plain",
            )


# -----------------------------------------------------------------------------
# VIEW 3: Concept Extraction (Agent 2)
# -----------------------------------------------------------------------------
elif menu_choice == "Concept Extraction":
    st.markdown("### 💡 Agent 2: Concept-Extractor Agent")

    if not filename:
        st.warning("Please upload a paper from the sidebar.")
    else:
        if not effective_api_key:
            st.info("ℹ️ **Offline Mode Active:** Extracting key concepts and benchmarks using offline keyword & term frequency analysis.")

        button_label = "✨ Extract Concepts with Gemini" if effective_api_key else "⚡ Extract Concepts Offline"
        if st.button(button_label, key="btn_run_concept"):
            with st.spinner("Concept-Extractor Agent is analyzing paper architecture..."):
                res = route_and_execute(
                    user_query="Extract and explain all core concepts, methodology, algorithms, and quantitative results.",
                    vectorstore=vectorstore,
                    filename=filename,
                    api_key=effective_api_key,
                    full_text=full_text,
                    explicit_intent=INTENT_CONCEPTS,
                )
                st.session_state.agent_outputs["concepts"] = res

        concept_res = st.session_state.agent_outputs.get("concepts")
        if concept_res and concept_res.get("status") == "success":
            badge_class = "badge-success" if "gemini" in concept_res.get("engine", "") else "badge-amber"
            st.markdown(
                f"""
                <div class="section-card">
                    <span class="badge {badge_class}">{concept_res.get('label', 'Concepts')}</span>
                    <div style="margin-top: 14px; font-size: 1.05rem; line-height: 1.7;">
                        {concept_res['output']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_source_citations(concept_res.get("sources", []))


# -----------------------------------------------------------------------------
# VIEW 4: Presentation Generator (Agent 3)
# -----------------------------------------------------------------------------
elif menu_choice == "Presentation Generator":
    st.markdown("### 🖥️ Agent 3: Slide-Generator Agent")

    if not filename:
        st.warning("Please upload a paper from the sidebar.")
    else:
        if not effective_api_key:
            st.info("ℹ️ **Offline Mode Active:** Generating slide deck outline from extracted document sections.")

        button_label = "✨ Generate Slides with Gemini" if effective_api_key else "⚡ Generate Offline Slide Deck"
        if st.button(button_label, key="btn_run_slides"):
            with st.spinner("Slide-Generator Agent is composing slide deck with speaker notes..."):
                res = route_and_execute(
                    user_query="Generate an 8-slide conference presentation deck with slide titles, concise bullet points, and speaker notes.",
                    vectorstore=vectorstore,
                    filename=filename,
                    api_key=effective_api_key,
                    full_text=full_text,
                    explicit_intent=INTENT_SLIDES,
                )
                st.session_state.agent_outputs["slides"] = res

        slide_res = st.session_state.agent_outputs.get("slides")
        if slide_res and slide_res.get("status") == "success":
            badge_class = "badge-success" if "gemini" in slide_res.get("engine", "") else "badge-amber"
            st.markdown(
                f"""
                <div class="section-card">
                    <span class="badge {badge_class}">{slide_res.get('label', 'Slide Deck')}</span>
                    <div style="margin-top: 14px;">
                        {slide_res['output']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_source_citations(slide_res.get("sources", []))

            if PPTX_AVAILABLE:
                try:
                    pptx_stream = generate_pptx_deck(
                        slide_res["output"],
                        paper_title=os.path.splitext(filename)[0],
                    )
                    st.download_button(
                        label="📊 Download Presentation (.pptx)",
                        data=pptx_stream,
                        file_name=f"{os.path.splitext(filename)[0]}_Slides.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )
                except Exception as e:
                    st.warning(f"Could not compile PPTX file: {e}. You can copy the Markdown outline above.")


# -----------------------------------------------------------------------------
# VIEW 5: Viva Preparation (Agent 4)
# -----------------------------------------------------------------------------
elif menu_choice == "Viva Preparation":
    st.markdown("### 🎓 Agent 4: Viva Agent (10 Oral Defense Questions)")

    if not filename:
        st.warning("Please upload a paper from the sidebar.")
    else:
        if not effective_api_key:
            st.info("ℹ️ **Offline Mode Active:** Generating 10 oral exam defense questions and grounded answers.")

        button_label = "✨ Generate Viva Questions with Gemini" if effective_api_key else "⚡ Generate 10 Viva Questions Offline"
        if st.button(button_label, key="btn_run_viva"):
            with st.spinner("Viva Agent is preparing exam questions and evaluation criteria..."):
                res = route_and_execute(
                    user_query="Generate 10 university-level viva questions and answers across basic, intermediate, and advanced levels.",
                    vectorstore=vectorstore,
                    filename=filename,
                    api_key=effective_api_key,
                    full_text=full_text,
                    explicit_intent=INTENT_VIVA,
                )
                st.session_state.agent_outputs["viva"] = res

        viva_res = st.session_state.agent_outputs.get("viva")
        if viva_res and viva_res.get("status") == "success":
            badge_class = "badge-success" if "gemini" in viva_res.get("engine", "") else "badge-amber"
            st.markdown(
                f"""
                <div class="section-card">
                    <span class="badge {badge_class}">{viva_res.get('label', 'Viva Questions')}</span>
                    <div style="margin-top: 14px;">
                        {viva_res['output']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_source_citations(viva_res.get("sources", []))


# -----------------------------------------------------------------------------
# VIEW 6: Ask Your Paper (Interactive Chat with Intent Routing)
# -----------------------------------------------------------------------------
elif menu_choice == "Ask Your Paper (Interactive Chat)":
    st.markdown("### 💬 Ask Your Paper (Intent Detection & Grounded Q&A)")

    if not filename:
        st.warning("Please upload a paper from the sidebar to start asking questions.")
    else:
        st.markdown(
            "Type any question, command, or request below. The **Intent Detection Router** will automatically "
            "classify your intent and dispatch your query to the appropriate agent or execute Grounded Q&A."
        )

        doc_memory = get_session_memory(st.session_state.session_id, filename)

        col_clear1, col_clear2 = st.columns([6, 1])
        with col_clear2:
            if st.button("🧹 Clear Chat"):
                doc_memory.clear()
                st.session_state.chat_history = []
                st.rerun()

        user_input = st.text_input(
            "Enter your question or request:",
            placeholder="e.g. 'What is the core breakthrough?', 'Summarize this', or 'Generate 8 slides'",
        )

        if st.button("Send Query", key="btn_send_chat") and user_input:
            with st.spinner("Router is classifying intent and executing pipeline..."):
                response = route_and_execute(
                    user_query=user_input,
                    vectorstore=vectorstore,
                    filename=filename,
                    memory=doc_memory,
                    api_key=effective_api_key,
                    full_text=full_text,
                )

                st.session_state.chat_history = [t.to_dict() for t in doc_memory.get_recent_history()]

        if st.session_state.chat_history:
            st.markdown("#### Conversation History")
            for turn in reversed(st.session_state.chat_history):
                if turn["role"] == "user":
                    st.markdown(
                        f"""
                        <div style="background-color: #e0f2fe; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                            <strong>👤 You:</strong> {turn['content']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    agent_badge = f"<span class='badge'>{turn.get('agent', 'Assistant')}</span>" if turn.get('agent') else ""
                    st.markdown(
                        f"""
                        <div class="qa-box">
                            {agent_badge}
                            <div style="margin-top: 6px; color: #0f172a; line-height: 1.6;">
                                {turn['content']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if turn.get("sources"):
                        render_source_citations(turn["sources"])


# -----------------------------------------------------------------------------
# VIEW 7: Extracted Raw Text
# -----------------------------------------------------------------------------
elif menu_choice == "Extracted Raw Text":
    st.markdown("### 📄 Extracted Document Text (PyMuPDF)")

    if not filename:
        st.warning("No PDF uploaded yet.")
    else:
        st.info(
            f"Showing raw text extracted via PyMuPDF from '{filename}' "
            f"({st.session_state.pdf_data['page_count']} pages, {st.session_state.pdf_data['word_count']:,} words)."
        )
        st.text_area(
            "Raw Extracted Text",
            value=st.session_state.pdf_data["text"],
            height=500,
            disabled=True,
        )
