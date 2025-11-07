import os
import re
import pickle
import streamlit as st
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

st.set_page_config(page_title="ATS + CV Analyzer", layout="wide")

# Load resources
@st.cache_resource
def load_model_and_encoder():
    try:
        model = pickle.load(open("resume_classifier.pkl", "rb"))
        encoder = SentenceTransformer("./resume_embedding_model")
        return model, encoder
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

model, encoder = load_model_and_encoder()

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Extract text from PDF
def extract_text(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + " "
        if text.strip():
            return text.strip()
    except:
        pass
    
    try:
        images = convert_from_path(path)
        for img in images:
            text += pytesseract.image_to_string(img) + " "
    except:
        pass
    
    return text.strip()

# Clean text for model
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9+#.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ML-based ATS scoring using trained model
def calculate_ats_score(resume_text, job_desc, model, encoder):
    if not job_desc.strip():
        return 0, [], [], "No job description provided"
    
    # Clean texts
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_desc)
    
    # Get embeddings
    resume_emb = encoder.encode([resume_clean])[0]
    jd_emb = encoder.encode([jd_clean])[0]
    
    # Calculate cosine similarity between resume and JD
    import numpy as np
    similarity = np.dot(resume_emb, jd_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(jd_emb))
    similarity_score = (similarity + 1) / 2 * 100  # Normalize to 0-100
    
    # Predict job category from resume
    resume_category = model.predict([resume_emb])[0]
    
    # Predict job category from JD
    jd_category = model.predict([jd_emb])[0]
    
    # Category match bonus
    category_match = (resume_category == jd_category)
    category_bonus = 20 if category_match else 0
    
    # Keyword overlap for transparency
    stop_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'will', 
                  'your', 'our', 'are', 'you', 'can', 'all', 'not', 'but', 'what', 'there'}
    
    jd_words = set(re.findall(r'\b[a-z+#]{3,}\b', jd_clean)) - stop_words
    resume_words = set(re.findall(r'\b[a-z+#]{3,}\b', resume_clean)) - stop_words
    
    matched = list(jd_words.intersection(resume_words))[:20]
    missing = list(jd_words - resume_words)[:15]
    
    # Final ATS score: weighted combination
    final_score = (similarity_score * 0.7) + (category_bonus * 0.3)
    final_score = min(final_score, 100)
    
    analysis = f"Resume Category: {resume_category} | JD Category: {jd_category} | Match: {'✓' if category_match else '✗'}"
    
    return round(final_score, 2), matched, missing, analysis

# UI
st.title("🎯 AI-Powered CV Analyzer + ATS Score")

with st.sidebar:
    st.header("📋 Instructions")
    st.write("1. Upload your resume (PDF)")
    st.write("2. Paste job description")
    st.write("3. Get AI evaluation & ATS score")
    
    if model is None:
        st.error("⚠️ Model not loaded. Run train_model.py first!")

resume_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("📝 Paste Job Description / Role Requirements", height=200, 
                        placeholder="Paste the complete job description here...")

if resume_file and model and encoder:
    # Save and extract
    with open("temp_resume.pdf", "wb") as f:
        f.write(resume_file.getbuffer())
    
    resume_text = extract_text("temp_resume.pdf")
    
    if not resume_text:
        st.error("❌ Could not extract text from PDF. Try a different file.")
        st.stop()
    
    # Show extracted text preview
    with st.expander("📄 View Extracted Resume Text"):
        st.text(resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Get AI Evaluation", use_container_width=True):
            if not api_key:
                st.warning("⚠️ Google API key not configured. Add GOOGLE_API_KEY to .env file.")
            else:
                with st.spinner("Analyzing resume..."):
                    prompt = f"""Analyze this resume and provide:
1. Overall assessment
2. Key strengths
3. Areas for improvement
4. Skill recommendations
5. Match with job requirements (if provided)

Resume:
{resume_text}

Job Requirements:
{job_desc if job_desc else 'Not provided'}
"""
                    try:
                        result = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
                        st.markdown("### 📋 AI Evaluation")
                        st.write(result.text)
                    except Exception as e:
                        st.error(f"AI evaluation failed: {e}")
    
    with col2:
        if st.button("✅ Calculate ATS Score", use_container_width=True):
            if not job_desc.strip():
                st.warning("⚠️ Please paste a job description to calculate ATS score.")
            else:
                with st.spinner("Calculating ML-based ATS score..."):
                    score, matched, missing, analysis = calculate_ats_score(resume_text, job_desc, model, encoder)
                    
                    st.markdown("### 📊 ML-Based ATS Compatibility Score")
                    st.progress(score / 100)
                    
                    # Color-coded score
                    if score >= 70:
                        st.success(f"**Score: {score}%** - Excellent Match! 🎉")
                    elif score >= 50:
                        st.warning(f"**Score: {score}%** - Good Match ✓")
                    else:
                        st.error(f"**Score: {score}%** - Needs Improvement ⚠️")
                    


elif resume_file and not model:
    st.error("⚠️ Model not loaded. Please run `python train_model.py` first.")
else:
    st.info("👆 Upload a resume PDF to begin analysis")
    
    with st.expander("ℹ️ How ML-Based ATS Scoring Works"):
        st.write("""
        **ML-Based ATS Score** uses trained machine learning models to evaluate resume-job fit.
        
        **Scoring Components:**
        1. **Semantic Similarity (70%)**: Uses AI embeddings to measure how similar your resume content is to the job description
        2. **Category Match (30%)**: Checks if your resume category matches the job category predicted by ML model
        
        **Score Ranges:**
        - 70-100%: Excellent match - High chance of passing ATS
        - 50-69%: Good match - Likely to pass with minor improvements
        - Below 50%: Needs improvement - Add more relevant keywords
        
        **Advantages over Traditional ATS:**
        - Understands context and meaning, not just keywords
        - Detects semantic similarity even with different wording
        - Uses trained model on 2,484 real resumes
        - More accurate job category prediction
        
        **Tips to Improve:**
        1. Align your experience with job requirements
        2. Use industry-standard terminology
        3. Ensure your skills match the job category
        4. Include relevant technical and soft skills
        """)

# Footer
st.markdown("---")
st.caption("🤖 ML-Powered ATS Scoring | Built with Streamlit, Google Gemini AI & SentenceTransformers")
