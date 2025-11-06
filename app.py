import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import pickle

# Must be first Streamlit command
st.set_page_config(page_title="CV Evaluation Tool", layout="wide")

# Load trained model and vectorizer on startup
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("resume_model.pkl", "rb"))
        vectorizer = pickle.load(open("resume_vectorizer.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        return None, None

resume_model, resume_vectorizer = load_model()

# Initialize environment configuration
load_dotenv()

# Setup Google Gemini AI
gemini_token = os.getenv("GOOGLE_API_KEY")
if not gemini_token:
    st.error("GOOGLE_API_KEY not configured. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=gemini_token)

# Function to parse document content
def parse_cv_document(document_location):
    content = ""
    try:
        # Attempt direct content parsing
        with pdfplumber.open(document_location) as doc:
            for doc_page in doc.pages:
                current_content = doc_page.extract_text()
                if current_content:
                    content += current_content

        if content.strip():
            return content.strip()
    except Exception as error:
        print(f"Primary parsing method failed: {error}")

    # Use OCR for scanned documents
    print("Switching to OCR processing for scanned document.")
    try:
        page_images = convert_from_path(document_location)
        for img in page_images:
            ocr_content = pytesseract.image_to_string(img)
            content += ocr_content + "\n"
    except Exception as error:
        print(f"OCR processing failed: {error}")

    return content.strip()

# Function to process CV evaluation
def process_cv_evaluation(cv_content, role_requirements=None):
    if not cv_content:
        return {"error": "CV content is required for evaluation."}
    
    ai_processor = genai.GenerativeModel("gemini-2.0-flash")

    
    evaluation_query = f"""
    You are a professional CV assessment specialist. Analyze the provided CV and deliver:
    
    - Overall evaluation of candidate profile
    - Current skills and competencies identified
    - Areas requiring enhancement
    - Recommended training programs and platforms
    - Key strengths and areas for improvement

    CV Content:
    {cv_content}
    """

    if role_requirements:
        evaluation_query += f"""
        Furthermore, assess this CV against the following role specifications:
        
        Role Specifications:
        {role_requirements}
        
        Identify candidate strengths and gaps relative to the specified position requirements.
        """

    ai_output = ai_processor.generate_content(evaluation_query)

    evaluation_result = ai_output.text.strip()
    return evaluation_result


# Streamlit application interface

# Application header
st.title("AI-Powered CV Assessment Platform")
st.write("Evaluate your CV and compare it with position requirements using Google Gemini AI.")

# Model Status
with st.sidebar:
    st.header("🤖 Model Status")
    
    if resume_model and resume_vectorizer:
        st.success("✅ Model is trained and ready!")
        st.info("📊 Ready for CV analysis and ATS scoring")
    else:
        st.error("❌ No trained model found")
        st.warning("Please run: python train_model.py")
        st.info("Model required for job prediction and ATS scoring")

left_column, right_column = st.columns(2)
with left_column:
    document_upload = st.file_uploader("Upload your CV document (PDF)", type=["pdf"])
with right_column:
    position_specs = st.text_area("Enter Position Requirements:", placeholder="Paste the position requirements here...")

if document_upload is not None:
    st.success("CV document uploaded successfully!")
else:
    st.warning("Please upload a CV document in PDF format.")


st.markdown("<div style= 'padding-top: 10px;'></div>", unsafe_allow_html=True)
if document_upload:
    # Store uploaded document locally for processing
    with open("uploaded_cv_document.pdf", "wb") as file_handler:
        file_handler.write(document_upload.getbuffer())
    # Parse content from document
    cv_data = parse_cv_document("uploaded_cv_document.pdf")

    # AI Job Category Prediction using trained model
    if resume_model and resume_vectorizer:
        predicted_role = resume_model.predict(resume_vectorizer.transform([cv_data]))[0]
        st.write(f"🧠 Predicted Job Role: **{predicted_role}**")
        
        col_eval, col_ats = st.columns(2)
        
        with col_eval:
            if st.button("Evaluate CV"):
                with st.spinner("Processing CV evaluation..."):
                    try:
                        evaluation_output = process_cv_evaluation(cv_data, position_specs)
                        st.success("Evaluation completed successfully!")
                        st.write(evaluation_output)
                    except Exception as error:
                        st.error(f"Evaluation process failed: {error}")
        
        with col_ats:
            if st.button("Calculate ATS Score"):
                confidence = max(resume_model.predict_proba(resume_vectorizer.transform([cv_data]))[0]) * 100
                st.subheader("📊 ATS Compatibility Results")
                st.progress(confidence / 100)
                st.write(f"**ATS Score:** {confidence:.2f}% for `{predicted_role}`")
                st.write("✅ **Analysis:** Based on trained ML model classification")
    else:
        if st.button("Evaluate CV"):
            with st.spinner("Processing CV evaluation..."):
                try:
                    evaluation_output = process_cv_evaluation(cv_data, position_specs)
                    st.success("Evaluation completed successfully!")
                    st.write(evaluation_output)
                except Exception as error:
                    st.error(f"Evaluation process failed: {error}")
        
        st.info("💡 Run 'python train_model.py' to enable ATS scoring")

#Application footer
st.markdown("---")
st.markdown("""<p style= 'text-align: center;' >Built with <b>Streamlit</b> and <b>Google Gemini AI</b></p>""", unsafe_allow_html=True)
