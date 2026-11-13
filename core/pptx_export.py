"""
Presentation Export Utility for PaperPilot-AI.
Generates an academic, styled PowerPoint (.pptx) deck from slide outlines.
"""

import io
import re
from typing import List, Dict, Any, Optional

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False


def parse_markdown_slides(slide_markdown: str) -> List[Dict[str, Any]]:
    """Parses markdown slides into structured slide objects."""
    slides = []
    # Match slide blocks like "### Slide 1: ..." or "Slide 1: ..."
    slide_blocks = re.split(r"(?:\r?\n)(?=###?\s+Slide\s+\d+:)", slide_markdown, flags=re.IGNORECASE)
    
    if len(slide_blocks) <= 1:
        # Fallback split on generic headings
        slide_blocks = re.split(r"(?:\r?\n)(?=###?\s+)", slide_markdown)

    for block in slide_blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue

        title = "Research Presentation"
        bullets = []
        speaker_notes = ""

        # Find title
        for idx, line in enumerate(lines):
            if re.match(r"^###?\s+", line):
                clean_title = re.sub(r"^###?\s+(?:Slide\s+\d+:\s*)?", "", line).strip()
                if clean_title:
                    title = clean_title
                break

        # Find speaker notes
        notes_match = re.search(r"\*{0,2}(?:🎙️\s*)?Speaker Notes:?\*{0,2}\s*(.+)$", block, flags=re.DOTALL | re.IGNORECASE)
        if notes_match:
            speaker_notes = notes_match.group(1).strip()
            # Clean up notes text
            speaker_notes = re.sub(r"---.*$", "", speaker_notes).strip()

        # Extract bullets (lines starting with -, *, or numbered)
        for line in lines:
            if re.match(r"^[-*•]\s+", line):
                bullet_txt = re.sub(r"^[-*•]\s+", "", line).strip()
                if not re.search(r"Speaker Notes", bullet_txt, re.I):
                    bullets.append(bullet_txt)
            elif re.match(r"^\d+\.\s+", line) and not re.match(r"^###?\s+Slide", line):
                bullet_txt = re.sub(r"^\d+\.\s+", "", line).strip()
                if not re.search(r"Speaker Notes", bullet_txt, re.I):
                    bullets.append(bullet_txt)

        if bullets or len(lines) > 1:
            slides.append({
                "title": title,
                "bullets": bullets if bullets else ["Key research findings and methodology."],
                "notes": speaker_notes,
            })

    return slides


def generate_pptx_deck(slide_markdown: str, paper_title: str = "Research Paper Presentation") -> io.BytesIO:
    """Creates a styled, professional academic PowerPoint presentation."""
    if not PPTX_AVAILABLE:
        raise RuntimeError("python-pptx is not installed in the Python environment.")

    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 Widescreen layout
    prs.slide_height = Inches(7.5)

    # Academic Colors (Navy and White/Sky theme)
    PRIMARY_COLOR = RGBColor(2, 132, 199)    # #0284c7 light-blue accent
    DARK_HEADER = RGBColor(15, 23, 42)       # #0f172a slate navy
    BODY_COLOR = RGBColor(51, 65, 85)        # #334155 slate gray
    ACCENT_BG = RGBColor(240, 249, 255)      # #f0f9ff light sky background

    blank_layout = prs.slide_layouts[6]  # Blank slide

    parsed_slides = parse_markdown_slides(slide_markdown)

    # 1. Title Slide
    title_slide = prs.slides.add_slide(blank_layout)
    
    # Title Box
    title_box = title_slide.shapes.add_textbox(Inches(1.2), Inches(2.2), Inches(11.0), Inches(2.5))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "✈️ PaperPilot-AI"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = PRIMARY_COLOR

    p2 = tf.add_paragraph()
    p2.text = paper_title
    p2.font.size = Pt(36)
    p2.font.bold = True
    p2.font.color.rgb = DARK_HEADER

    p3 = tf.add_paragraph()
    p3.text = "Automated Academic Slide Deck • Synthesized via Agentic RAG"
    p3.font.size = Pt(16)
    p3.font.color.rgb = BODY_COLOR

    # 2. Content Slides
    for slide_data in parsed_slides:
        slide = prs.slides.add_slide(blank_layout)

        # Header Title Banner
        title_box = slide.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(1.2))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = slide_data["title"]
        p_title.font.size = Pt(28)
        p_title.font.bold = True
        p_title.font.color.rgb = DARK_HEADER

        # Content Bullet Box
        content_box = slide.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(11.3), Inches(4.5))
        tf_body = content_box.text_frame
        tf_body.word_wrap = True

        for idx, bullet in enumerate(slide_data["bullets"]):
            p_bullet = tf_body.paragraphs[0] if idx == 0 else tf_body.add_paragraph()
            p_bullet.text = f"•  {bullet}"
            p_bullet.font.size = Pt(18)
            p_bullet.font.color.rgb = BODY_COLOR
            p_bullet.space_after = Pt(14)

        # Speaker Notes
        if slide_data.get("notes"):
            notes_slide = slide.notes_slide
            notes_tf = notes_slide.notes_text_frame
            notes_tf.text = slide_data["notes"]

    # Save to memory buffer
    output_stream = io.BytesIO()
    prs.save(output_stream)
    output_stream.seek(0)
    return output_stream
