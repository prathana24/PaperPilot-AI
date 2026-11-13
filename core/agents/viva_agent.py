"""
Agent 4: Viva Agent
Generates 10 research-paper-specific viva-voce questions spanning basic,
intermediate, and advanced difficulty with comprehensive model answers using LCEL.
Incorporates academic few-shot exemplar prompting for rigorous viva answer formatting.
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
4. Follow the format demonstrated in the few-shot academic examples below.

FORMAT SPECIFICATION:
### Question X: [Clear, targeted oral defense question]
- **Difficulty Level:** [Basic / Intermediate / Advanced]
- **Ideal Model Answer:** [Clear, authoritative answer grounded in the paper]
- **💡 Examiner Focus Tip:** [What key insight or depth the candidate must demonstrate]
---
"""

# Few-shot exemplars demonstrating expected academic viva questioning and response rigor
FEW_SHOT_EXAMPLE_HUMAN = """Document: benchmark_example_study.pdf

RETRIEVED PAPER EXCERPTS:
[Excerpt from Page 2]:
We evaluate ResNet-50 and Vision Transformer architectures on ImageNet-1K. The primary bottleneck addressed is quadratic computational complexity in multi-head self-attention, which scales as O(N^2) with sequence length N. Our LinAttention module achieves linear complexity O(N) by decomposing the softmax kernel into separable low-rank projections. LinAttention yields a 2.4x throughput speedup on NVIDIA A100 GPUs while maintaining 81.2% top-1 accuracy (within 0.3% of full self-attention).

CUSTOM EXAM PREFERENCES:
Demonstrate the standard viva defense questioning style."""

FEW_SHOT_EXAMPLE_AI = """### Question 1: What core computational bottleneck motivated the development of the LinAttention module?
- **Difficulty Level:** Basic
- **Ideal Model Answer:** The authors targeted the quadratic computational complexity of standard multi-head self-attention, which scales as O(N^2) with respect to input sequence length N. This quadratic scaling creates severe memory and throughput bottlenecks when processing high-resolution visual inputs.
- **💡 Examiner Focus Tip:** Ensure the candidate clearly articulates the asymptotic scaling limitation (O(N^2)) rather than offering vague qualitative claims about speed.
---

### Question 2: How does the proposed linear projection mechanism mathematically preserve attention quality while accelerating throughput?
- **Difficulty Level:** Advanced
- **Ideal Model Answer:** LinAttention replaces the standard softmax matrix multiplication by decomposing the kernel into separable low-rank linear projections. This reduces time and memory complexity to O(N). Empirically, this decomposition achieves a 2.4x throughput speedup on NVIDIA A100 GPUs while restricting top-1 accuracy drop to just 0.3% (81.2% vs. 81.5% for standard self-attention).
- **💡 Examiner Focus Tip:** Look for candidate understanding of the theoretical trade-off between exact softmax normalization and separable low-rank kernel approximation."""

VIVA_HUMAN_PROMPT = """Document: {filename}

RETRIEVED PAPER EXCERPTS:
{context}

CUSTOM EXAM PREFERENCES:
{user_query}

Now generate all 10 university-level viva questions and model answers following the exact format and rigor demonstrated in the few-shot examples above."""


def create_viva_chain(llm):
    """Builds the few-shot LCEL chain: ChatPromptTemplate | llm | StrOutputParser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", VIVA_SYSTEM_PROMPT),
        ("human", FEW_SHOT_EXAMPLE_HUMAN),
        ("ai", FEW_SHOT_EXAMPLE_AI),
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
    """Executes the Viva Agent with live few-shot LCEL or offline generation."""
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

    # 2. LIVE GEMINI LCEL PATH (WITH FEW-SHOT EXEMPLARS)
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
            "engine": "langchain_lcel_gemini_fewshot",
            "label": "Gemini-Powered Few-Shot Viva Examination (LCEL)"
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
