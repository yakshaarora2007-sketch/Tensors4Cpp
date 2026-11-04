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
    # Replace unicode symbols with PDF-standard ASCII approximations
    replacements = {
        '—': '--', '–': '-', '“': '"', '”': '"', '‘': "'", '’': "'",
        '×': 'x', '·': '*', '≈': '~=', '≠': '!=', '≤': '<=', '≥': '>=',
        '→': '->', '←': '<-', '⇒': '=>', '⇔': '<=>', '∈': 'in',
        'mathbb{R}': 'R', 'mathbb{N}': 'N', 'mathbb{Z}': 'Z',
        'mathcal{T}': 'T', 'mathcal{A}': 'A', 'mathcal{B}': 'B', 'mathcal{C}': 'C',
        'mathcal{I}': 'I', 'mathcal{K}': 'K', 'mathcal{M}': 'M',
        'mathbf{A}': 'A', 'mathbf{B}': 'B', 'mathbf{C}': 'C', 'mathbf{W}': 'W',
        'mathbf{X}': 'X', 'mathbf{Y}': 'Y', 'mathbf{v}': 'v', 'mathbf{i}': 'i',
        'mathbf{S}': 'S', 'odot': '(x)', 'pm': '+/-', 'le': '<=', 'ge': '>=',
        'prod': 'PROD', 'sum': 'SUM', 'infty': 'inf', 'delta': 'd', 'alpha': 'a',
        'epsilon': 'eps', 'Omega': 'Offset', 'times': 'x', 'quad': '  ',
        'dots': '...', 'text': '', '{': '', '}': '', '\\': '', '$': ''
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
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
            # Header (Pages 2+)
            if page_num > 1:
                page.set_stroke_color(0.8, 0.82, 0.85)
                page.set_line_width(0.75)
                page.draw_line(MARGIN_LEFT, PAGE_HEIGHT - 38.0, PAGE_WIDTH - MARGIN_RIGHT, PAGE_HEIGHT - 38.0)
                page.draw_text("Design and Implementation of a Lightweight Multidimensional Tensor Library in C++",
                               MARGIN_LEFT, PAGE_HEIGHT - 34.0, "F1", 8.0, 0.4, 0.4, 0.45)
            # Footer (All pages)
            page.set_stroke_color(0.8, 0.82, 0.85)
            page.set_line_width(0.75)
            page.draw_line(MARGIN_LEFT, 42.0, PAGE_WIDTH - MARGIN_RIGHT, 42.0)
            page.draw_text("MiniTensor Technical Research Paper | Systems Engineering",
                           MARGIN_LEFT, 30.0, "F1", 8.0, 0.45, 0.45, 0.5)
            page_str = f"Page {page_num} of {total_pages}"
            str_w = string_width_helvetica(page_str, 8.0)
            page.draw_text(page_str, PAGE_WIDTH - MARGIN_RIGHT - str_w, 30.0, "F2", 8.0, 0.3, 0.3, 0.35)

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
    abstract_lines = []
    in_table = False
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Handle Code Blocks
        if line.startswith("```"):
            if in_code_block:
                # End of code block: render it
                in_code_block = False
                font_size = 7.5
                line_height = 10.5
                padding = 8.0
                block_height = len(code_lines) * line_height + 2 * padding
                doc.check_space(block_height + 10.0)
                
                # Draw background box
                doc.current_page.set_fill_color(0.96, 0.965, 0.975)
                doc.current_page.set_stroke_color(0.85, 0.87, 0.90)
                doc.current_page.set_line_width(0.75)
                doc.current_page.draw_rect(MARGIN_LEFT, doc.cursor_y - block_height, USABLE_WIDTH, block_height, fill=True, stroke=True)
                
                # Render code text
                y_text = doc.cursor_y - padding - font_size
                for cl in code_lines:
                    cl_sanitized = sanitize_pdf_text(cl)
                    if len(cl_sanitized) > 95:
                        cl_sanitized = cl_sanitized[:92] + "..."
                    doc.current_page.draw_text(cl_sanitized, MARGIN_LEFT + 8.0, y_text, "F4", font_size, 0.15, 0.20, 0.25)
                    y_text -= line_height
                doc.cursor_y -= block_height + 10.0
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
            # Check if this is a separator line like |---|---|
            if set(line.replace("|", "").replace(" ", "").replace(":", "")) == {"-"}:
                i += 1
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            table_rows.append(cells)
            # Look ahead to see if table continues
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                continue
            else:
                # End of table: render table
                num_cols = max(len(r) for r in table_rows)
                col_width = USABLE_WIDTH / num_cols
                row_height = 18.0
                total_table_height = len(table_rows) * row_height
                doc.check_space(total_table_height + 14.0)

                table_top = doc.cursor_y
                for r_idx, row in enumerate(table_rows):
                    y_row = table_top - (r_idx + 1) * row_height
                    is_header = (r_idx == 0)
                    # Background
                    if is_header:
                        doc.current_page.set_fill_color(0.18, 0.28, 0.45)
                        doc.current_page.draw_rect(MARGIN_LEFT, y_row, USABLE_WIDTH, row_height, fill=True, stroke=False)
                    elif r_idx % 2 == 1:
                        doc.current_page.set_fill_color(0.97, 0.98, 0.99)
                        doc.current_page.draw_rect(MARGIN_LEFT, y_row, USABLE_WIDTH, row_height, fill=True, stroke=False)

                    # Cell Borders
                    doc.current_page.set_stroke_color(0.82, 0.85, 0.88)
                    doc.current_page.set_line_width(0.5)
                    doc.current_page.draw_rect(MARGIN_LEFT, y_row, USABLE_WIDTH, row_height, fill=False, stroke=True)

                    # Text
                    for c_idx, cell in enumerate(row):
                        cell_clean = sanitize_pdf_text(cell).replace("**", "")
                        c_x = MARGIN_LEFT + c_idx * col_width + 4.0
                        c_y = y_row + 5.0
                        font = "F2" if is_header else "F1"
                        r_col = 1.0 if is_header else 0.15
                        g_col = 1.0 if is_header else 0.15
                        b_col = 1.0 if is_header else 0.20
                        # Truncate if too long
                        max_chars = int(col_width / 5.2)
                        if len(cell_clean) > max_chars:
                            cell_clean = cell_clean[:max_chars - 2] + ".."
                        doc.current_page.draw_text(cell_clean, c_x, c_y, font, 7.5, r_col, g_col, b_col)

                doc.cursor_y -= total_table_height + 12.0
                table_rows = []
                i += 1
                continue

        # Handle Headings
        if line.startswith("# "):
            title_text = line[2:].strip()
            doc.check_space(80.0)
            doc.cursor_y -= 10.0
            doc.current_page.draw_text(title_text, MARGIN_LEFT, doc.cursor_y, "F2", 18.0, 0.10, 0.18, 0.35)
            doc.cursor_y -= 14.0
            doc.current_page.set_stroke_color(0.18, 0.28, 0.45)
            doc.current_page.set_line_width(2.0)
            doc.current_page.draw_line(MARGIN_LEFT, doc.cursor_y, MARGIN_LEFT + 120.0, doc.cursor_y)
            doc.cursor_y -= 16.0
            i += 1
            continue

        if line.startswith("## "):
            h2_text = line[3:].strip()
            doc.check_space(45.0)
            doc.cursor_y -= 14.0
            doc.current_page.draw_text(h2_text, MARGIN_LEFT, doc.cursor_y, "F2", 12.5, 0.12, 0.22, 0.40)
            doc.cursor_y -= 6.0
            doc.current_page.set_stroke_color(0.85, 0.88, 0.92)
            doc.current_page.set_line_width(0.75)
            doc.current_page.draw_line(MARGIN_LEFT, doc.cursor_y, MARGIN_LEFT + USABLE_WIDTH, doc.cursor_y)
            doc.cursor_y -= 10.0
            i += 1
            continue

        if line.startswith("### "):
            h3_text = line[4:].strip()
            doc.check_space(30.0)
            doc.cursor_y -= 10.0
            doc.current_page.draw_text(h3_text, MARGIN_LEFT, doc.cursor_y, "F2", 10.5, 0.18, 0.28, 0.45)
            doc.cursor_y -= 12.0
            i += 1
            continue

        if line.startswith("#### "):
            h4_text = line[5:].strip()
            doc.check_space(25.0)
            doc.cursor_y -= 8.0
            doc.current_page.draw_text(h4_text, MARGIN_LEFT, doc.cursor_y, "F2", 9.5, 0.25, 0.30, 0.38)
            doc.cursor_y -= 11.0
            i += 1
            continue

        # Handle Metadata lines (Author, Date, etc.)
        if line.startswith("**Author:**") or line.startswith("**Affiliation:**") or line.startswith("**Date:**") or line.startswith("**Document Type:**"):
            doc.check_space(16.0)
            clean_line = sanitize_pdf_text(line).replace("**", "")
            doc.current_page.draw_text(clean_line, MARGIN_LEFT, doc.cursor_y, "F1", 9.0, 0.35, 0.38, 0.42)
            doc.cursor_y -= 13.0
            i += 1
            continue

        if line == "---":
            doc.check_space(14.0)
            doc.cursor_y -= 4.0
            doc.current_page.set_stroke_color(0.85, 0.88, 0.92)
            doc.current_page.set_line_width(0.75)
            doc.current_page.draw_line(MARGIN_LEFT, doc.cursor_y, MARGIN_LEFT + USABLE_WIDTH, doc.cursor_y)
            doc.cursor_y -= 12.0
            i += 1
            continue

        # Handle Abstract Heading & Content
        if line.strip() == "## Abstract":
            doc.check_space(120.0)
            doc.cursor_y -= 8.0
            # Read subsequent lines until next heading or divider
            abstract_text = []
            i += 1
            while i < len(lines) and not lines[i].startswith("---") and not lines[i].startswith("## "):
                if lines[i].strip():
                    abstract_text.append(lines[i].strip())
                i += 1
            full_abs = " ".join(abstract_text)
            wrapped_abs = wrap_text(full_abs, 8.5, USABLE_WIDTH - 24.0)
            box_h = len(wrapped_abs) * 12.0 + 34.0

            # Draw Callout Box for Abstract
            doc.current_page.set_fill_color(0.96, 0.97, 0.99)
            doc.current_page.set_stroke_color(0.20, 0.35, 0.60)
            doc.current_page.set_line_width(1.0)
            doc.current_page.draw_rect(MARGIN_LEFT, doc.cursor_y - box_h, USABLE_WIDTH, box_h, fill=True, stroke=True)
            
            # Abstract Title
            doc.current_page.draw_text("ABSTRACT", MARGIN_LEFT + 12.0, doc.cursor_y - 18.0, "F2", 9.5, 0.15, 0.25, 0.45)
            
            # Abstract Body
            y_abs = doc.cursor_y - 32.0
            for al in wrapped_abs:
                doc.current_page.draw_text(al, MARGIN_LEFT + 12.0, y_abs, "F3", 8.5, 0.20, 0.22, 0.25)
                y_abs -= 12.0
            doc.cursor_y -= box_h + 14.0
            continue

        # Handle Math / Display Equation Blocks ($$ ... $$)
        if line.startswith("$$") and line.endswith("$$") and len(line) > 4:
            eq_text = line[2:-2].strip()
            eq_clean = sanitize_pdf_text(eq_text)
            doc.check_space(28.0)
            doc.cursor_y -= 6.0
            # Indented math display
            doc.current_page.draw_text(eq_clean, MARGIN_LEFT + 36.0, doc.cursor_y, "F7", 9.5, 0.10, 0.15, 0.30)
            doc.cursor_y -= 16.0
            i += 1
            continue

        # Handle List items
        if line.strip().startswith("- ") or line.strip().startswith("* "):
            bullet_text = line.strip()[2:]
            wrapped = wrap_text(bullet_text, 8.8, USABLE_WIDTH - 18.0)
            doc.check_space(len(wrapped) * 12.5 + 4.0)
            for l_idx, wl in enumerate(wrapped):
                if l_idx == 0:
                    doc.current_page.draw_text("*", MARGIN_LEFT + 4.0, doc.cursor_y, "F2", 9.0, 0.25, 0.35, 0.50)
                doc.current_page.draw_text(wl, MARGIN_LEFT + 14.0, doc.cursor_y, "F1", 8.8, 0.18, 0.18, 0.20)
                doc.cursor_y -= 12.5
            i += 1
            continue

        # Numbered list items: e.g. 1. , 2.
        m_num = re.match(r"^(\d+\.)\s+(.*)", line.strip())
        if m_num:
            prefix = m_num.group(1)
            rest = m_num.group(2)
            wrapped = wrap_text(rest, 8.8, USABLE_WIDTH - 24.0)
            doc.check_space(len(wrapped) * 12.5 + 4.0)
            for l_idx, wl in enumerate(wrapped):
                if l_idx == 0:
                    doc.current_page.draw_text(prefix, MARGIN_LEFT + 2.0, doc.cursor_y, "F2", 8.8, 0.20, 0.25, 0.35)
                doc.current_page.draw_text(wl, MARGIN_LEFT + 20.0, doc.cursor_y, "F1", 8.8, 0.18, 0.18, 0.20)
                doc.cursor_y -= 12.5
            i += 1
            continue

        # Blank Line
        if not line.strip():
            doc.cursor_y -= 6.0
            i += 1
            continue

        # Standard Paragraph
        wrapped_para = wrap_text(line, 8.8, USABLE_WIDTH)
        doc.check_space(len(wrapped_para) * 12.5 + 4.0)
        for pl in wrapped_para:
            doc.current_page.draw_text(pl, MARGIN_LEFT, doc.cursor_y, "F1", 8.8, 0.15, 0.15, 0.18)
            doc.cursor_y -= 12.5
        doc.cursor_y -= 3.0
        i += 1

    # Decorate all pages with headers, footers, page counts
    doc.add_page_decorations()

    # Compile into binary PDF 1.4 file
    write_pdf_file(doc.pages, output_pdf_path)

def write_pdf_file(pages: List[PDFCanvas], output_path: str):
    objects: List[bytes] = []
    offsets: List[int] = []

    # Helper to add an object
    def add_object(content: bytes) -> int:
        obj_num = len(objects) + 1
        objects.append(content)
        return obj_num

    # Fonts: F1 (Helvetica), F2 (Helvetica-Bold), F3 (Helvetica-Oblique), F4 (Courier), F7 (Times-Bold)
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

    # Resources dictionary
    resources = (
        f"<< /Font << /F1 {id_f1} 0 R /F2 {id_f2} 0 R /F3 {id_f3} 0 R /F4 {id_f4} 0 R /F7 {id_f7} 0 R >> >>"
    ).encode('latin-1')
    id_resources = add_object(resources)

    # Page objects and Contents
    page_ids = []
    # Reserve page object IDs
    start_page_id = len(objects) + 1
    num_pages = len(pages)
    
    # We will generate contents first, then pages, then pages root, then catalog
    # Let's placeholder page IDs
    for p_idx in range(num_pages):
        page_ids.append(start_page_id + p_idx * 2)

    pages_root_id = start_page_id + num_pages * 2

    for p_idx, page in enumerate(pages):
        # 1. Content stream
        raw_stream = "\n".join(page.commands).encode('latin-1')
        compressed_stream = zlib.compress(raw_stream)
        stream_obj = (
            f"<< /Length {len(compressed_stream)} /Filter /FlateDecode >>\nstream\n"
        ).encode('latin-1') + compressed_stream + b"\nendstream"
        
        id_page = add_object(b"") # Placeholder
        id_content = add_object(stream_obj)
        
        # Now fill page object
        page_dict = (
            f"<< /Type /Page /Parent {pages_root_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources {id_resources} 0 R /Contents {id_content} 0 R >>"
        ).encode('latin-1')
        objects[id_page - 1] = page_dict

    # Pages root
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_root = f"<< /Type /Pages /Kids [ {kids_str} ] /Count {num_pages} >>".encode('latin-1')
    id_pages_root = add_object(pages_root)

    # Catalog
    catalog = f"<< /Type /Catalog /Pages {id_pages_root} 0 R >>".encode('latin-1')
    id_catalog = add_object(catalog)

    # Assemble complete PDF byte buffer
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

    # XRef table
    xref_start = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode('latin-1'))
    for off in xref_offsets:
        pdf_bytes.extend(f"{off:010d} 00000 n \n".encode('latin-1'))

    # Trailer
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

