import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import pdfplumber

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
    You are a seasoned talent acquisition specialist with technical expertise across various domains including Data Science, Analytics, DevOps, Machine Learning, AI Engineering, Full Stack Development, Big Data, Marketing Analytics, HR Management, and Software Development. Your objective is to assess the provided CV.
    Deliver a comprehensive evaluation regarding candidate suitability for relevant positions. Identify existing competencies and recommend skill enhancements along with suitable training programs. Emphasize strengths and areas for improvement.

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

st.set_page_config(page_title="CV Evaluation Tool", layout="wide")
# Application header
st.title("AI-Powered CV Assessment Platform")
st.write("Evaluate your CV and compare it with position requirements using Google Gemini AI.")

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

    if st.button("Evaluate CV"):
        with st.spinner("Processing CV evaluation..."):
            try:
                # Process CV evaluation
                evaluation_output = process_cv_evaluation(cv_data, position_specs)
                st.success("Evaluation completed successfully!")
                st.write(evaluation_output)
            except Exception as error:
                st.error(f"Evaluation process failed: {error}")

#Application footer
st.markdown("---")
st.markdown("""<p style= 'text-align: center;' >Built with <b>Streamlit</b> and <b>Google Gemini AI</b></p>""", unsafe_allow_html=True)
