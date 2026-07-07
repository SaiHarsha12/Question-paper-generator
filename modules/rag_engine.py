from langchain_ollama import OllamaLLM
import random
import re

llm = OllamaLLM(model="llama3", streaming=False)

def generate_exam_question(subtopic, bloom_level, question_type, difficulty):
    if isinstance(subtopic, list):
        subtopic = ", ".join(subtopic)
    
    subtopic_lower = str(subtopic).lower()
    
    diff_context = {
        "Easy": "Use straightforward terminology. Ask for basic definitions and fundamental working.",
        "Moderate": "Ask for comparisons, internal mechanisms, and logical derivations.",
        "Hard": "Frame complex scenarios, critical evaluations, and multi-step problems."
    }[difficulty]
    

    if any(k in subtopic_lower for k in ["algorithm", "method", "process", "working"]):
        verb = random.choice(["Explain the working of", "Illustrate the mechanism of", "Demonstrate the steps of"])
        requirement = "Provide a neat diagram and pseudocode."
    elif any(k in subtopic_lower for k in ["types", "challenges", "issues", "approaches"]):
        verb = random.choice(["List and elaborate on", "Categorize and explain", "Classify the various"])
        requirement = "Discuss with suitable real-world examples."
    elif any(k in subtopic_lower for k in ["difference", "comparison", "vs"]):
        verb = "Differentiate between"
        requirement = "Use a comparison table for clarity."
    else:
        verb = random.choice(["Describe", "Elaborate on", "Discuss in detail", "Explain the significance of"])
        requirement = "Explain its applications in modern systems."

    prompt = f"""
You are an expert BMSCE Professor. 
Frame ONE clear 10-mark question for a {difficulty} level exam.

SUBTOPIC: {subtopic}
BLOOM'S LEVEL: {bloom_level}
DIFFICULTY: {difficulty} ({diff_context})

STRICT INSTRUCTIONS:
1. Start the question with: "{verb}".
2. {requirement}
3. If the topic is mathematical (Bayes, SVM, PCA, K-Means), frame a {difficulty} NUMERICAL problem.
4. If it involves a system/model, ask for a neat DIAGRAM.
5. Do NOT use introductory text or "Here is your question".
6. Ensure the depth is sufficient for 10 marks.

OUTPUT ONLY THE QUESTION TEXT.
"""

    response = llm.generate([prompt])
    return response.generations[0][0].text.strip()