import pdfplumber
import re

def get_unit_concepts(pdf_path):
    text_all = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_all += t + "\n"

    pattern = r"(UNIT[\s\-–—]*([0-9]+))"
    matches = list(re.finditer(pattern, text_all, re.I))

    units = {}
    for i in range(len(matches)):
        start = matches[i].end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text_all)
        num = matches[i].group(2)
        content = text_all[start:end]

       
        content = re.sub(r"\d+\s*Hrs?\.?", "", content, flags=re.I)
        content = re.sub(r"(Text Books|Reference Books|Course Outcomes).*", "", content, flags=re.S | re.I)
        
       
        subtopics = re.split(r'[,;\n]', content)
        
    
        clean_subtopics = []
        for s in subtopics:
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > 3 and not s.lower().startswith("introduction"):
                clean_subtopics.append(s)

        units[f"UNIT-{num}"] = clean_subtopics 
    return units