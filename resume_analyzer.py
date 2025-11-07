# import os
# import pdfplumber
# import pytesseract
# from pdf2image import convert_from_path
# import google.generativeai as genai
# from dotenv import load_dotenv


# def parse_document_content(file_location):
#     """Parse content from text-based and scanned document files."""
#     extracted_content = ""

#     try:
#         with pdfplumber.open(file_location) as document:
#             for doc_page in document.pages:
#                 current_text = doc_page.extract_text()
#                 if current_text:
#                     extracted_content += current_text + "\n"

#         if extracted_content.strip():
#             return extracted_content.strip()

#     except Exception as error:
#         print(f"[!] Primary parsing failed: {error}")

#     print("[*] Switching to OCR method...")
#     try:
#         page_images = convert_from_path(file_location)
#         for image_obj in page_images:
#             extracted_content += pytesseract.image_to_string(image_obj) + "\n"

#     except Exception as error:
#         print(f"[!] OCR parsing failed: {error}")

#     return extracted_content.strip()


# def evaluate_cv_content(cv_data, position_requirements=None):
#     """Evaluate CV content using Google Gemini AI."""
#     if not cv_data:
#         return "No CV content available for evaluation."

#     evaluation_prompt = f"""
# You are a seasoned talent acquisition specialist with expertise in technical recruitment.
# Examine the provided CV and deliver:

# - Comprehensive assessment
# - Current competencies identified
# - Areas requiring enhancement
# - Recommended training programs (include learning platforms)
# - Key advantages and limitations

# CV Content:
# {cv_data}
# """

#     if position_requirements:
#         evaluation_prompt += f"""

# Analyze alignment between the CV and the following position requirements.

# Position Requirements:
# {position_requirements}
# """

#     ai_model = genai.GenerativeModel("gemini-1.5-flash")
#     ai_response = ai_model.generate_content(evaluation_prompt)

#     return ai_response.text.strip()


# if __name__ == "__main__":
#     load_dotenv()
#     gemini_key = os.getenv("GOOGLE_API_KEY")

#     if not gemini_key:
#         raise ValueError("GOOGLE_API_KEY not configured in .env file. Please copy .env.example to .env and add your API key.")

#     genai.configure(api_key=gemini_key)

#     # Get document path from user input
#     document_path = input("Provide the path to your CV document (PDF): ").strip()
    
#     if not os.path.exists(document_path):
#         print(f"Error: Document '{document_path}' not located.")
#         exit(1)
    
#     cv_content = parse_document_content(document_path)

#     if not cv_content:
#         print("Error: Unable to parse document content.")
#         exit(1)

#     print("\n===== Parsed CV Content =====\n")
#     print(cv_content)

#     # Optional position requirements
#     position_desc = input("\nProvide position requirements (optional, press Enter to skip): ").strip()
#     position_desc = position_desc if position_desc else None

#     print("\n===== CV Evaluation Results =====\n")
#     print(evaluate_cv_content(cv_content, position_desc))
import os
import re
import pickle
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Optional: tweak UI
st.set_page_config(page_title="ATS + CV Analyzer", layout="wide")
st.title("AI-Powered CV Analyzer + ATS Score")

# 1) Configuration (adjustable)
MODEL_PATH = "resume_classifier.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_OUTPUT = "./resume_embedding_model"
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

# ATS
ATS_SKILLS = [
    "python","java","c++","c#","sql","mysql","mongodb","django","flask","nlp",
    "machine learning","deep learning","data analysis","pandas","numpy","react",
    "node","aws","docker","power bi","excel","html","css","javascript"
]

# 2) Helpers
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def detect_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9+#.\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def extract_text_from_pdf_or_image(file_location: str) -> str:
    extracted = ""

    # Primary: text extraction via pdfplumber if available
    try:
        with pdfplumber.open(file_location) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    extracted += content + "\n"
        if extracted.strip():
            return extracted.strip()
    except Exception:
        pass

    # Fallback: OCR on PDF pages
    try:
        pages = convert_from_path(file_location)
        for img in pages:
            if 'pytesseract' in globals():
                extracted += pytesseract.image_to_string(img) + "\n"
        return extracted.strip()
    except Exception:
        return extracted.strip()

def calculate_ats(resume_text: str, jd_text: str, skills=ATS_SKILLS):
    resume = resume_text.lower()
    jd = jd_text.lower()
    matched = [s for s in skills if s in resume and s in jd]
    score = (len(matched) / len(skills)) * 100
    return round(score, 2), matched

# 3) Load resources (cache-like behavior implemented via Streamlit state)
# We avoid using old caching decorators; instead we rely on Streamlit's session state for persistence
def get_resources():
    # Embedding model
    if "encoder" not in st.session_state:
        st.session_state.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        st.session_state.embedding_name = EMBEDDING_MODEL_NAME
    # Classifier model
    if "model" not in st.session_state:
        if not os.path.exists(MODEL_PATH):
            st.error("Model not found. Train or provide resume_classifier.pkl at path: {}".format(MODEL_PATH))
            st.stop()
        st.session_state.model = load_model(MODEL_PATH)
    return st.session_state.model, st.session_state.encoder

model, encoder = get_resources()

# 4) Gemini setup (optional)
load_dotenv()
api_key = os.getenv(GOOGLE_API_KEY_ENV)
if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass

# 5) UI inputs
with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_desc = st.text_area("Paste Job Description / Role Requirements", height=200)

if uploaded is not None:
    tmp_path = "temp_resume.pdf"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Parse resume
    raw_text = extract_text_from_pdf_or_image(tmp_path)
    if not raw_text:
        st.error("Failed to extract text from the uploaded document.")
        st.stop()

    cleaned = clean_text(raw_text)

    # Embedding + classification
    emb = encoder.encode([cleaned])
    predicted_role = model.predict(emb)[0]

    st.write(f"🧠 Predicted Job Category: `{predicted_role}`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Evaluate CV"):
            prompt = f"Evaluate the following CV:\n{raw_text}\nJob Requirements:\n{job_desc}"
            if api_key:
                try:
                    result = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text
                    st.write(result)
                except Exception as e:
                    st.warning(f"AI evaluation failed: {e}")
            else:
                st.info("Gemini API key not configured. Provide GOOGLE_API_KEY to enable AI evaluation.")
    with col2:
        if st.button("✅ Calculate ATS Score"):
            if not job_desc:
                st.warning("Please paste a job description to compute ATS score.")
            else:
                score, matched = calculate_ats(raw_text, job_desc)
                st.progress(min(max(score, 0), 100) / 100)
                st.write(f"ATS Score: {int(score)}%")
                st.write("Matched Skills:", matched)
else:
    st.info("Upload a resume PDF to begin.")

