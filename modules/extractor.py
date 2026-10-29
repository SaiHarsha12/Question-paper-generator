"""Syllabus and Course Structure Extractor for Academic Curriculum Analysis.

Course: Natural Language Processing for LLM (24AM7PCLLM)
Institution: B.M.S. College of Engineering (BMSCE)
Department: Artificial Intelligence & Machine Learning

This module extracts structured syllabus content from academic PDFs:
1. Heading-based Unit extraction ('UNIT 1' / 'MODULE I').
2. Table-style Course Plan extraction (e.g., Week | Class | Unit | Topic).
3. Course Outcomes (CO1..CO5) extraction with descriptions from tables and text.
4. Guaranteed backward-compatibility for existing question paper pipelines.
"""

from typing import Dict, List, Any, Optional, Union
import re
import pdfplumber

ROMAN_TO_INT: Dict[str, str] = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"
}


def extract_course_structure(pdf_path: str) -> Dict[str, Any]:
    """Extract complete course curriculum structure including Units, Subtopics, and COs.

    Supports both heading-structured syllabi (UNIT-1 ...) and tabular lesson plans
    (columns containing Week | Class | Unit | Topic).

    Args:
        pdf_path: Path to the syllabus or course plan PDF.

    Returns:
        Dictionary containing:
        {
            "units": Dict[str, List[str]],  # e.g. {"UNIT-1": ["Topic A", ...], ...}
            "cos": Dict[str, str]           # e.g. {"CO1": "Apply protocols...", ...}
        }
    """
    text_all = ""
    all_tables: List[List[List[Optional[str]]]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_all += t + "\n"
            tables = page.extract_tables()
            if tables:
                all_tables.extend(tables)

    # 1. Extract Course Outcomes (COs) from tables and text
    cos = _extract_course_outcomes(text_all, all_tables)

    # 2. Extract Units from table-based course plans
    table_units = _extract_table_course_plan(all_tables)

    # 3. Extract Units from text headings (UNIT 1..5, MODULE I..V)
    heading_units = _extract_heading_units(text_all)

    # Prioritize: if heading units found, use them; if table plan has units, merge/use them
    final_units: Dict[str, List[str]] = {}
    if heading_units:
        final_units = heading_units
        # If table plan has additional detailed topics for those units, supplement them
        for u_key, topics in table_units.items():
            if u_key not in final_units or not final_units[u_key]:
                final_units[u_key] = topics
    elif table_units:
        final_units = table_units

    # Sort units naturally (UNIT-1, UNIT-2, ...)
    sorted_units = dict(sorted(
        final_units.items(),
        key=lambda item: int(re.search(r"\d+", item[0]).group()) if re.search(r"\d+", item[0]) else 99
    ))

    return {
        "units": sorted_units,
        "cos": cos
    }


def _extract_heading_units(text: str) -> Dict[str, List[str]]:
    """Extract units from document text using UNIT/MODULE heading patterns."""
    pattern = r"(?:UNIT|MODULE)[\s\-–—]*([0-9]+|[IVXLCDM]+)\b"
    matches = list(re.finditer(pattern, text, re.I))

    units: Dict[str, List[str]] = {}
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_num = matches[i].group(1).strip()
        num = ROMAN_TO_INT.get(raw_num.lower(), raw_num)
        content = text[start:end]

        # Strip hours and textbook / CO references
        content = re.sub(r"\d+\s*Hrs?\.?", "", content, flags=re.I)
        content = re.sub(r"(Text\s*Books|Reference\s*Books|Course\s*Outcomes).*", "", content, flags=re.S | re.I)

        subtopics = re.split(r"[,;\n]", content)
        clean_subtopics: List[str] = []
        for s in subtopics:
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > 3 and not s.lower().startswith("introduction"):
                clean_subtopics.append(s)

        if not clean_subtopics:
            fallback = [line.strip() for line in content.splitlines() if len(line.strip()) > 3]
            clean_subtopics = fallback if fallback else [f"Unit {num} Concepts"]

        units[f"UNIT-{num}"] = clean_subtopics

    return units


def _extract_table_course_plan(tables: List[List[List[Optional[str]]]]) -> Dict[str, List[str]]:
    """Extract topics grouped by unit from table-style lesson plans.

    Recognizes formats like: Week | Class | Unit | Topic / Content.
    """
    table_units: Dict[str, List[str]] = {}

    for table in tables:
        if not table or len(table) < 2:
            continue

        # Look for header row
        unit_col_idx = -1
        topic_col_idx = -1

        for row_idx, row in enumerate(table[:3]):
            clean_row = [str(c or "").strip().lower() for c in row]
            for col_idx, cell in enumerate(clean_row):
                if any(k in cell for k in ["unit", "module", "u.no"]):
                    unit_col_idx = col_idx
                elif any(k in cell for k in ["topic", "portion", "content", "syllabus", "lesson", "plan"]):
                    topic_col_idx = col_idx

            if unit_col_idx != -1 and topic_col_idx != -1:
                start_row = row_idx + 1
                break
        else:
            # Fallback heuristic: check if any column contains single digits (1..5)
            # and another column has longer strings
            continue

        current_unit: Optional[str] = None
        for row in table[start_row:]:
            if not row or len(row) <= max(unit_col_idx, topic_col_idx):
                continue

            unit_cell = str(row[unit_col_idx] or "").strip()
            topic_cell = str(row[topic_col_idx] or "").strip()

            # Check if this row specifies a new unit number
            unit_match = re.search(r"\b([1-9]|10|[IVX]+)\b", unit_cell, re.I)
            if unit_match:
                val = unit_match.group(1).lower()
                current_unit = ROMAN_TO_INT.get(val, val)

            if current_unit and topic_cell and len(topic_cell) > 3:
                # Clean up topic text
                topic_clean = re.sub(r"\s+", " ", topic_cell)
                topic_clean = re.sub(r"(?i)\b\d+\s*hrs?\.?\b", "", topic_clean).strip()
                if topic_clean and len(topic_clean) > 3:
                    u_key = f"UNIT-{current_unit}"
                    if u_key not in table_units:
                        table_units[u_key] = []
                    if topic_clean not in table_units[u_key]:
                        table_units[u_key].append(topic_clean)

    return table_units


def _extract_course_outcomes(text: str, tables: List[List[List[Optional[str]]]]) -> Dict[str, str]:
    """Extract Course Outcomes (CO1..CO5) and their descriptions."""
    cos: Dict[str, str] = {}

    # 1. Search in tables (common in VTU / BMSCE syllabi)
    for table in tables:
        if not table:
            continue
        for row in table:
            if not row or len(row) < 2:
                continue
            first_cell = str(row[0] or "").strip()
            second_cell = str(row[1] or "").strip()

            co_match = re.match(r"^(CO\s*[1-6])\b", first_cell, re.I)
            if co_match and second_cell and len(second_cell) > 5:
                co_key = co_match.group(1).replace(" ", "").upper()
                # Clean up description
                desc = re.sub(r"\s+", " ", second_cell).strip()
                cos[co_key] = desc
            elif not co_match and len(row) >= 3:
                # Try column 1 as CO key and column 2 as description
                co_match2 = re.match(r"^(CO\s*[1-6])\b", second_cell, re.I)
                third_cell = str(row[2] or "").strip()
                if co_match2 and third_cell and len(third_cell) > 5:
                    co_key = co_match2.group(1).replace(" ", "").upper()
                    cos[co_key] = re.sub(r"\s+", " ", third_cell).strip()

    # 2. Text-based regex search fallback if tables didn't yield COs
    if len(cos) < 2:
        pattern = r"(?i)\b(CO\s*[1-6])\b\s*[:\-–—]\s*([^\n\r]+(?:\n(?!\s*(?:CO\s*[1-6]|PO\s*\d|Text\s*Book))[^\n\r]+)*)"
        for match in re.finditer(pattern, text):
            co_key = match.group(1).replace(" ", "").upper()
            desc = re.sub(r"\s+", " ", match.group(2)).strip()
            if co_key not in cos and len(desc) > 5:
                cos[co_key] = desc

    return dict(sorted(cos.items()))


def get_unit_concepts(
    pdf_path: str,
    return_full: bool = False
) -> Union[Dict[str, List[str]], Dict[str, Any]]:
    """Backward-compatible wrapper to extract course units or full structure.

    Args:
        pdf_path: Path to the syllabus PDF.
        return_full: If True, returns {'units': {...}, 'cos': {...}}.
            If False (default), returns the legacy unit dictionary:
            {'UNIT-1': [...], 'UNIT-2': [...]}.

    Returns:
        Legacy unit dictionary or full dictionary depending on return_full.
    """
    structure = extract_course_structure(pdf_path)
    if return_full:
        return structure
    return structure["units"]
