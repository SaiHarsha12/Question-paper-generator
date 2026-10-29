import pdfplumber
import re

ROMAN_TO_INT = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"
}

def get_unit_concepts(pdf_path):
    text_all = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_all += t + "\n"

    pattern = r"(?:UNIT|MODULE)[\s\-–—]*([0-9]+|[IVXLCDM]+)\b"
    matches = list(re.finditer(pattern, text_all, re.I))

    units = {}
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text_all)
        raw_num = matches[i].group(1).strip()
        num = ROMAN_TO_INT.get(raw_num.lower(), raw_num)
        content = text_all[start:end]

        content = re.sub(r"\d+\s*Hrs?\.?", "", content, flags=re.I)
        content = re.sub(r"(Text Books|Reference Books|Course Outcomes).*", "", content, flags=re.S | re.I)
        
        subtopics = re.split(r'[,;\n]', content)
        
        clean_subtopics = []
        for s in subtopics:
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > 3 and not s.lower().startswith("introduction"):
                clean_subtopics.append(s)

        if not clean_subtopics:
            fallback = [line.strip() for line in content.splitlines() if len(line.strip()) > 3]
            clean_subtopics = fallback if fallback else [f"Unit {num} Concepts"]

        units[f"UNIT-{num}"] = clean_subtopics 
    return units
