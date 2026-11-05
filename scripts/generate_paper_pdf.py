"""
MiniTensor Research Paper PDF Generator
Generates a publication-quality PDF version of docs/mini_tensor_library_paper.md
Zero external dependencies - uses pure Python standard library.
"""

import os
import re
import sys
import zlib
from typing import List, Tuple, Dict, Any

# Standard Letter Page dimensions in points (72 points per inch)
PAGE_WIDTH = 612.0
PAGE_HEIGHT = 792.0
MARGIN_LEFT = 54.0
MARGIN_RIGHT = 54.0
MARGIN_TOP = 54.0
MARGIN_BOTTOM = 54.0
USABLE_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
USABLE_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

# Helvetica AFM character metrics (widths in 1/1000th of em)
HELVETICA_WIDTHS = {
    32: 278, 33: 278, 34: 355, 35: 556, 36: 556, 37: 889, 38: 667, 39: 191,
    40: 333, 41: 333, 42: 389, 43: 584, 44: 278, 45: 333, 46: 278, 47: 278,
    48: 556, 49: 556, 50: 556, 51: 556, 52: 556, 53: 556, 54: 556, 55: 556,
    56: 556, 57: 556, 58: 278, 59: 278, 60: 584, 61: 584, 62: 584, 63: 556,
    64: 1015, 65: 667, 66: 667, 67: 722, 68: 722, 69: 667, 70: 611, 71: 778,
    72: 722, 73: 278, 74: 500, 75: 667, 76: 556, 77: 833, 78: 722, 79: 778,
    80: 667, 81: 778, 82: 722, 83: 667, 84: 611, 85: 722, 86: 667, 87: 944,
    88: 667, 89: 667, 90: 611, 91: 278, 92: 278, 93: 278, 94: 469, 95: 556,
    96: 333, 97: 556, 98: 556, 99: 500, 100: 556, 101: 556, 102: 278, 103: 556,
    104: 556, 105: 222, 106: 222, 107: 500, 108: 222, 109: 833, 110: 556,
    111: 556, 112: 556, 113: 556, 114: 333, 115: 500, 116: 278, 117: 556,
    118: 500, 119: 722, 120: 500, 121: 500, 122: 500, 123: 334, 124: 260,
    125: 334, 126: 584
}

def string_width_helvetica(text: str, font_size: float) -> float:
    total = 0
    for ch in text:
        code = ord(ch)
        width = HELVETICA_WIDTHS.get(code, 556)
        total += width
    return (total / 1000.0) * font_size

def string_width_courier(text: str, font_size: float) -> float:
    return len(text) * 0.6 * font_size

def sanitize_pdf_text(text: str) -> str:
    # 1. Unicode character replacements
    unicode_map = {
        '—': '--', '–': '-', '“': '"', '”': '"', '‘': "'", '’': "'",
        '×': 'x', '·': '*', '≈': '~=', '≠': '!=', '≤': '<=', '≥': '>=',
        '→': '->', '←': '<-', '⇒': '=>', '⇔': '<=>', '∈': 'in', '∇': 'grad',
        '∂': 'd', '…': '...', '±': '+/-', '•': '*'
    }
    for k, v in unicode_map.items():
        text = text.replace(k, v)

    # 2. Markdown links: [text](url) -> text if text == url else text (url)
    def _link_repl(m):
        t, u = m.group(1), m.group(2)
        return t if t == u else f"{t} ({u})"
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _link_repl, text)

    # 3. LaTeX replacements via regex
    text = re.sub(r'\\(?:mathbb|mathbf|mathcal|text)\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\le\b', '<=', text)
    text = re.sub(r'\\ge\b', '>=', text)
    text = re.sub(r'\\times\b', 'x', text)
    text = re.sub(r'\\cdot\b', '*', text)
    text = re.sub(r'\\pm\b', '+/-', text)
    text = re.sub(r'\\odot\b', '(x)', text)
    text = re.sub(r'\\prod\b', 'PROD', text)
    text = re.sub(r'\\sum\b', 'SUM', text)
    text = re.sub(r'\\to\b', '->', text)
    text = re.sub(r'\\leftarrow\b', '<-', text)
    text = re.sub(r'\\approx\b', '~=', text)
    text = re.sub(r'\\neq\b', '!=', text)
    text = re.sub(r'\\in\b', 'in', text)
    text = re.sub(r'\\alpha\b', 'alpha', text)
    text = re.sub(r'\\beta\b', 'beta', text)
    text = re.sub(r'\\gamma\b', 'gamma', text)
    text = re.sub(r'\\delta\b', 'delta', text)
    text = re.sub(r'\\epsilon\b', 'eps', text)
    text = re.sub(r'\\nabla\b', 'grad', text)
    text = re.sub(r'\\partial\b', 'd', text)
    text = re.sub(r'\\infty\b', 'inf', text)
    text = re.sub(r'\\quad\b', '  ', text)
    text = re.sub(r'\\dots\b', '...', text)
    text = re.sub(r'\\Omega\b', 'Offset', text)

    # Clean LaTeX math delimiters $
    text = text.replace('$', '')
    # Strip backslashes
    text = text.replace('\\', '')

    # Filter to printable ASCII
    clean_chars = []
    for c in text:
        if 32 <= ord(c) <= 126:
            clean_chars.append(c)
        elif c == '\n':
            clean_chars.append(' ')
        else:
            clean_chars.append(' ')
    return "".join(clean_chars)

def escape_pdf_string(text: str) -> str:
    res = []
    for c in text:
        if c == '\\':
            res.append('\\\\')
        elif c == '(':
            res.append('\\(')
        elif c == ')':
            res.append('\\)')
        else:
            res.append(c)
    return "".join(res)

class PDFCanvas:
    def __init__(self):
        self.commands: List[str] = []

    def set_fill_color(self, r: float, g: float, b: float):
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} rg")

    def set_stroke_color(self, r: float, g: float, b: float):
        self.commands.append(f"{r:.3f} {g:.3f} {b:.3f} RG")

    def set_line_width(self, width: float):
        self.commands.append(f"{width:.2f} w")

    def draw_rect(self, x: float, y: float, w: float, h: float, fill: bool = True, stroke: bool = False):
        op = "B" if fill and stroke else ("f" if fill else "S")
        self.commands.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}")

    def draw_line(self, x1: float, y1: float, x2: float, y2: float):
        self.commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def draw_text(self, text: str, x: float, y: float, font: str, size: float, r: float = 0.0, g: float = 0.0, b: float = 0.0):
        clean = escape_pdf_string(sanitize_pdf_text(text))
        self.commands.append(f"q {r:.3f} {g:.3f} {b:.3f} rg BT /{font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({clean}) Tj ET Q")

class FlowDocument:
    def __init__(self):
        self.pages: List[PDFCanvas] = []
        self.current_page: PDFCanvas = None
        self.cursor_y: float = 0.0
        self.new_page()

    def new_page(self):
        self.current_page = PDFCanvas()
        self.pages.append(self.current_page)
        self.cursor_y = PAGE_HEIGHT - MARGIN_TOP

    def check_space(self, required_height: float):
        if self.cursor_y - required_height < MARGIN_BOTTOM:
            self.new_page()

    def add_page_decorations(self):
        total_pages = len(self.pages)
        for i, page in enumerate(self.pages):
            page_num = i + 1
            # Running Header (Pages 2+)
            if page_num > 1:
                page.set_stroke_color(0.80, 0.82, 0.86)
                page.set_line_width(0.75)
                page.draw_line(MARGIN_LEFT, PAGE_HEIGHT - 38.0, PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 38.0)
                page.draw_text("Design and Implementation of a Lightweight Multidimensional Tensor Library in C++",
                               MARGIN_LEFT, PAGE_HEIGHT - 34.0, "F1", 8.0, 0.40, 0.42, 0.46)
            # Running Footer (All pages)
            page.set_stroke_color(0.80, 0.82, 0.86)
            page.set_line_width(0.75)
            page.draw_line(MARGIN_LEFT, 42.0, PAGE_WIDTH - MARGIN_RIGHT, 42.0)
            page.draw_text("MiniTensor Technical Research Paper | Systems Engineering",
                           MARGIN_LEFT, 30.0, "F1", 8.0, 0.42, 0.44, 0.48)
            page_str = f"Page {page_num} of {total_pages}"
            str_w = string_width_helvetica(page_str, 8.0)
            page.draw_text(page_str, PAGE_WIDTH - MARGIN_RIGHT - str_w, 30.0, "F2", 8.0, 0.30, 0.32, 0.36)

def wrap_text(text: str, font_size: float, max_width: float, is_code: bool = False) -> List[str]:
    text = sanitize_pdf_text(text).strip()
    if not text:
        return []
    words = text.split(' ')
    lines = []
    current_line = []
    for word in words:
        if not word:
            continue
        test_line = " ".join(current_line + [word])
        w = string_width_courier(test_line, font_size) if is_code else string_width_helvetica(test_line, font_size)
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def render_markdown_to_pdf(md_content: str, output_pdf_path: str):
    doc = FlowDocument()
    lines = md_content.splitlines()
    in_code_block = False
    code_lines = []
    in_abstract = False
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Handle Code Blocks
        if line.startswith("```"):
            if in_code_block:
                in_code_block = False
                font_size = 7.2
                line_height = 10.0
                padding = 6.0
                
                # Render code lines with page-splitting awareness
                remaining_code = list(code_lines)
                while remaining_code:
                    avail_height = doc.cursor_y - MARGIN_BOTTOM - 16.0
                    lines_fit = int((avail_height - 2 * padding) / line_height)
                    if lines_fit < 3 and avail_height < USABLE_HEIGHT - 40.0:
                        doc.new_page()
                        avail_height = doc.cursor_y - MARGIN_BOTTOM - 16.0
                        lines_fit = int((avail_height - 2 * padding) / line_height)
                    
                    chunk = remaining_code[:max(1, lines_fit)]
                    remaining_code = remaining_code[len(chunk):]
                    chunk_height = len(chunk) * line_height + 2 * padding
                    
                    # Background box
                    doc.current_page.set_fill_color(0.965, 0.970, 0.978)
                    doc.current_page.set_stroke_color(0.84, 0.86, 0.89)
                    doc.current_page.set_line_width(0.6)
                    doc.current_page.draw_rect(MARGIN_LEFT, doc.cursor_y - chunk_height, USABLE_WIDTH, chunk_height, fill=True, stroke=True)
                    
                    y_text = doc.cursor_y - padding - font_size
                    for cl in chunk:
                        cl_sanitized = sanitize_pdf_text(cl)
                        if len(cl_sanitized) > 105:
                            cl_sanitized = cl_sanitized[:102] + "..."
                        doc.current_page.draw_text(cl_sanitized, MARGIN_LEFT + 8.0, y_text, "F4", font_size, 0.15, 0.20, 0.25)
                        y_text -= line_height
                    
                    doc.cursor_y -= chunk_height + 8.0
                    if remaining_code:
                        doc.new_page()
                
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Handle Tables
        if line.startswith("|") and line.endswith("|"):
            if set(line.replace("|", "").replace(" ", "").replace(":", "")) == {"-"}:
                i += 1
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            table_rows.append(cells)
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                continue
            else:
                num_cols = max(len(r) for r in table_rows)
                col_width = USABLE_WIDTH / num_cols
                row_height = 17.0
                total_table_height = len(table_rows) * row_height
                
                if doc.cursor_y - total_table_height < MARGIN_BOTTOM:
                    if total_table_height <= USABLE_HEIGHT:
                        doc.new_page()

                table_top = doc.cursor_y
                for r_idx, row in enumerate(table_rows):
                    y_row = table_top - (r_idx + 1) * row_height
                    is_header = (r_idx == 0)
                    if is_header:
                        doc.current_page.set_fill_color(0.18, 0.28, 0.45)
                        doc.current_page.draw_rect(MARGIN_LEFT, y_row, USABLE_WIDTH, row_height, fill=True, stroke=False)
                    elif r_idx % 2 == 1:
                        doc.current_page.set_fill_color(0.965, 0.975, 0.985)
                        doc.current_page.draw_rect(MARGIN_LEFT, y_row, USABLE_WIDTH, row_height, fill=True, stroke=False)

                    doc.current_page.set_stroke_color(0.82, 0.85, 0.88)
                    doc.current_page.set_line_width(0.5)
                    doc.current_page.draw_rect(MARGIN_LEFT, y_row, USABLE_WIDTH, row_height, fill=False, stroke=True)

                    for c_idx, cell in enumerate(row):
                        cell_clean = sanitize_pdf_text(cell).replace("**", "")
                        c_x = MARGIN_LEFT + c_idx * col_width + 4.0
                        c_y = y_row + 4.5
                        font = "F2" if is_header else "F1"
                        r_col = 1.0 if is_header else 0.15
                        g_col = 1.0 if is_header else 0.15
                        b_col = 1.0 if is_header else 0.20
                        max_chars = int(col_width / 5.0)
                        if len(cell_clean) > max_chars:
                            cell_clean = cell_clean[:max_chars - 2] + ".."
                        doc.current_page.draw_text(cell_clean, c_x, c_y, font, 7.2, r_col, g_col, b_col)

                doc.cursor_y -= total_table_height + 10.0
                table_rows = []
                i += 1
                continue

        # Handle Headings
        if line.startswith("# "):
            title_text = line[2:].strip()
            doc.check_space(80.0)
            doc.cursor_y -= 8.0
            doc.current_page.draw_text(title_text, MARGIN_LEFT, doc.cursor_y, "F2", 17.0, 0.10, 0.18, 0.35)
            doc.cursor_y -= 12.0
            doc.current_page.set_stroke_color(0.18, 0.28, 0.45)
            doc.current_page.set_line_width(2.0)
            doc.current_page.draw_line(MARGIN_LEFT, doc.cursor_y, MARGIN_LEFT + 140.0, doc.cursor_y)
            doc.cursor_y -= 14.0
            i += 1
            continue

        if line.startswith("## "):
            h2_text = line[3:].strip()
            doc.check_space(42.0)
            doc.cursor_y -= 12.0
            doc.current_page.draw_text(h2_text, MARGIN_LEFT, doc.cursor_y, "F2", 12.0, 0.12, 0.22, 0.40)
            doc.cursor_y -= 5.0
            doc.current_page.set_stroke_color(0.85, 0.88, 0.92)
            doc.current_page.set_line_width(0.75)
            doc.current_page.draw_line(MARGIN_LEFT, doc.cursor_y, MARGIN_LEFT + USABLE_WIDTH, doc.cursor_y)
            doc.cursor_y -= 9.0
            i += 1
            continue

        if line.startswith("### "):
            h3_text = line[4:].strip()
            doc.check_space(28.0)
            doc.cursor_y -= 8.0
            doc.current_page.draw_text(h3_text, MARGIN_LEFT, doc.cursor_y, "F2", 10.0, 0.18, 0.28, 0.45)
            doc.cursor_y -= 10.0
            i += 1
            continue

        if line.startswith("#### "):
            h4_text = line[5:].strip()
            doc.check_space(24.0)
            doc.cursor_y -= 6.0
            doc.current_page.draw_text(h4_text, MARGIN_LEFT, doc.cursor_y, "F2", 9.2, 0.25, 0.30, 0.38)
            doc.cursor_y -= 9.0
            i += 1
            continue

        # Handle Metadata lines
        if any(line.startswith(prefix) for prefix in [
            "**Author:**", "**Affiliation:**", "**Repository:**",
            "**Date:**", "**Document Type:**", "**Version:**"
        ]):
            doc.check_space(15.0)
            clean_line = sanitize_pdf_text(line).replace("**", "")
            doc.current_page.draw_text(clean_line, MARGIN_LEFT, doc.cursor_y, "F1", 8.8, 0.30, 0.34, 0.40)
            doc.cursor_y -= 12.5
            i += 1
            continue

        if line == "---":
            doc.check_space(12.0)
            doc.cursor_y -= 3.0
            doc.current_page.set_stroke_color(0.85, 0.88, 0.92)
            doc.current_page.set_line_width(0.75)
            doc.current_page.draw_line(MARGIN_LEFT, doc.cursor_y, MARGIN_LEFT + USABLE_WIDTH, doc.cursor_y)
            doc.cursor_y -= 10.0
            i += 1
            continue

        # Handle Abstract Block
        if line.strip() == "## Abstract":
            doc.check_space(110.0)
            doc.cursor_y -= 6.0
            abstract_text = []
            i += 1
            while i < len(lines) and not lines[i].startswith("---") and not lines[i].startswith("## "):
                if lines[i].strip():
                    abstract_text.append(lines[i].strip())
                i += 1
            full_abs = " ".join(abstract_text)
            wrapped_abs = wrap_text(full_abs, 8.4, USABLE_WIDTH - 24.0)
            box_h = len(wrapped_abs) * 11.5 + 30.0

            doc.current_page.set_fill_color(0.965, 0.975, 0.99)
            doc.current_page.set_stroke_color(0.20, 0.35, 0.60)
            doc.current_page.set_line_width(1.0)
            doc.current_page.draw_rect(MARGIN_LEFT, doc.cursor_y - box_h, USABLE_WIDTH, box_h, fill=True, stroke=True)
            
            doc.current_page.draw_text("ABSTRACT", MARGIN_LEFT + 12.0, doc.cursor_y - 16.0, "F2", 9.0, 0.15, 0.25, 0.45)
            y_abs = doc.cursor_y - 28.0
            for al in wrapped_abs:
                doc.current_page.draw_text(al, MARGIN_LEFT + 12.0, y_abs, "F3", 8.4, 0.20, 0.22, 0.25)
                y_abs -= 11.5
            doc.cursor_y -= box_h + 12.0
            continue

        # Handle Display Equations ($$ ... $$)
        if line.startswith("$$") and line.endswith("$$") and len(line) > 4:
            eq_text = line[2:-2].strip()
            eq_clean = sanitize_pdf_text(eq_text)
            doc.check_space(26.0)
            doc.cursor_y -= 5.0
            doc.current_page.draw_text(eq_clean, MARGIN_LEFT + 32.0, doc.cursor_y, "F7", 9.2, 0.10, 0.15, 0.30)
            doc.cursor_y -= 14.0
            i += 1
            continue

        # Handle List Items
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            bullet_text = line.strip()[2:]
            wrapped = wrap_text(bullet_text, 8.5, USABLE_WIDTH - 18.0)
            doc.check_space(len(wrapped) * 12.0 + 3.0)
            for l_idx, wl in enumerate(wrapped):
                if l_idx == 0:
                    doc.current_page.draw_text("*", MARGIN_LEFT + 4.0, doc.cursor_y, "F2", 8.8, 0.25, 0.35, 0.50)
                doc.current_page.draw_text(wl, MARGIN_LEFT + 14.0, doc.cursor_y, "F1", 8.5, 0.18, 0.18, 0.20)
                doc.cursor_y -= 12.0
            i += 1
            continue

        # Numbered list items
        m_num = re.match(r"^(\d+\.)\s+(.*)", line.strip())
        if m_num:
            prefix = m_num.group(1)
            rest = m_num.group(2)
            wrapped = wrap_text(rest, 8.5, USABLE_WIDTH - 24.0)
            doc.check_space(len(wrapped) * 12.0 + 3.0)
            for l_idx, wl in enumerate(wrapped):
                if l_idx == 0:
                    doc.current_page.draw_text(prefix, MARGIN_LEFT + 2.0, doc.cursor_y, "F2", 8.5, 0.20, 0.25, 0.35)
                doc.current_page.draw_text(wl, MARGIN_LEFT + 20.0, doc.cursor_y, "F1", 8.5, 0.18, 0.18, 0.20)
                doc.cursor_y -= 12.0
            i += 1
            continue

        # Blank Line
        if not line.strip():
            doc.cursor_y -= 5.0
            i += 1
            continue

        # Standard Paragraph
        wrapped_para = wrap_text(line, 8.5, USABLE_WIDTH)
        doc.check_space(len(wrapped_para) * 12.0 + 3.0)
        for pl in wrapped_para:
            doc.current_page.draw_text(pl, MARGIN_LEFT, doc.cursor_y, "F1", 8.5, 0.15, 0.15, 0.18)
            doc.cursor_y -= 12.0
        doc.cursor_y -= 2.5
        i += 1

    # Decorate all pages with headers, footers, page counts
    doc.add_page_decorations()

    # Compile into binary PDF 1.4 file
    write_pdf_file(doc.pages, output_pdf_path)

def write_pdf_file(pages: List[PDFCanvas], output_path: str):
    objects: List[bytes] = []
    offsets: List[int] = []

    def add_object(content: bytes) -> int:
        obj_num = len(objects) + 1
        objects.append(content)
        return obj_num

    # Fonts
    font_f1 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    font_f2 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
    font_f3 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>"
    font_f4 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"
    font_f7 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold /Encoding /WinAnsiEncoding >>"

    id_f1 = add_object(font_f1)
    id_f2 = add_object(font_f2)
    id_f3 = add_object(font_f3)
    id_f4 = add_object(font_f4)
    id_f7 = add_object(font_f7)

    resources = (
        f"<< /Font << /F1 {id_f1} 0 R /F2 {id_f2} 0 R /F3 {id_f3} 0 R /F4 {id_f4} 0 R /F7 {id_f7} 0 R >> >>"
    ).encode('latin-1')
    id_resources = add_object(resources)

    page_ids = []
    start_page_id = len(objects) + 1
    num_pages = len(pages)
    
    for p_idx in range(num_pages):
        page_ids.append(start_page_id + p_idx * 2)

    pages_root_id = start_page_id + num_pages * 2

    for p_idx, page in enumerate(pages):
        raw_stream = "\n".join(page.commands).encode('latin-1')
        compressed_stream = zlib.compress(raw_stream)
        stream_obj = (
            f"<< /Length {len(compressed_stream)} /Filter /FlateDecode >>\nstream\n"
        ).encode('latin-1') + compressed_stream + b"\nendstream"
        
        id_page = add_object(b"") # Placeholder
        id_content = add_object(stream_obj)
        
        page_dict = (
            f"<< /Type /Page /Parent {pages_root_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources {id_resources} 0 R /Contents {id_content} 0 R >>"
        ).encode('latin-1')
        objects[id_page - 1] = page_dict

    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_root = f"<< /Type /Pages /Kids [ {kids_str} ] /Count {num_pages} >>".encode('latin-1')
    id_pages_root = add_object(pages_root)

    catalog = f"<< /Type /Catalog /Pages {id_pages_root} 0 R >>".encode('latin-1')
    id_catalog = add_object(catalog)

    pdf_bytes = bytearray()
    pdf_bytes.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    xref_offsets = []
    for idx, obj in enumerate(objects):
        obj_num = idx + 1
        offset = len(pdf_bytes)
        xref_offsets.append(offset)
        pdf_bytes.extend(f"{obj_num} 0 obj\n".encode('latin-1'))
        pdf_bytes.extend(obj)
        pdf_bytes.extend(b"\nendobj\n")

    xref_start = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode('latin-1'))
    for off in xref_offsets:
        pdf_bytes.extend(f"{off:010d} 00000 n \n".encode('latin-1'))

    pdf_bytes.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {id_catalog} 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode('latin-1')
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(pdf_bytes)

    print(f"Successfully generated {output_path} ({len(pdf_bytes)} bytes, {num_pages} pages).")

if __name__ == "__main__":
    paper_md_path = os.path.join(os.path.dirname(__file__), "..", "docs", "mini_tensor_library_paper.md")
    output_pdf = os.path.join(os.path.dirname(__file__), "..", "docs", "mini_tensor_library_paper.pdf")
    if not os.path.exists(paper_md_path):
        print(f"Error: {paper_md_path} not found.")
        sys.exit(1)
    with open(paper_md_path, "r", encoding="utf-8") as f:
        content = f.read()
    render_markdown_to_pdf(content, output_pdf)
