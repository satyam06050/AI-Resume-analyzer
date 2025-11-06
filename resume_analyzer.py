import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import google.generativeai as genai
from dotenv import load_dotenv


def parse_document_content(file_location):
    """Parse content from text-based and scanned document files."""
    extracted_content = ""

    try:
        with pdfplumber.open(file_location) as document:
            for doc_page in document.pages:
                current_text = doc_page.extract_text()
                if current_text:
                    extracted_content += current_text + "\n"

        if extracted_content.strip():
            return extracted_content.strip()

    except Exception as error:
        print(f"[!] Primary parsing failed: {error}")

    print("[*] Switching to OCR method...")
    try:
        page_images = convert_from_path(file_location)
        for image_obj in page_images:
            extracted_content += pytesseract.image_to_string(image_obj) + "\n"

    except Exception as error:
        print(f"[!] OCR parsing failed: {error}")

    return extracted_content.strip()


def evaluate_cv_content(cv_data, position_requirements=None):
    """Evaluate CV content using Google Gemini AI."""
    if not cv_data:
        return "No CV content available for evaluation."

    evaluation_prompt = f"""
You are a seasoned talent acquisition specialist with expertise in technical recruitment.
Examine the provided CV and deliver:

- Comprehensive assessment
- Current competencies identified
- Areas requiring enhancement
- Recommended training programs (include learning platforms)
- Key advantages and limitations

CV Content:
{cv_data}
"""

    if position_requirements:
        evaluation_prompt += f"""

Analyze alignment between the CV and the following position requirements.

Position Requirements:
{position_requirements}
"""

    ai_model = genai.GenerativeModel("gemini-1.5-flash")
    ai_response = ai_model.generate_content(evaluation_prompt)

    return ai_response.text.strip()


if __name__ == "__main__":
    load_dotenv()
    gemini_key = os.getenv("GOOGLE_API_KEY")

    if not gemini_key:
        raise ValueError("GOOGLE_API_KEY not configured in .env file. Please copy .env.example to .env and add your API key.")

    genai.configure(api_key=gemini_key)

    # Get document path from user input
    document_path = input("Provide the path to your CV document (PDF): ").strip()
    
    if not os.path.exists(document_path):
        print(f"Error: Document '{document_path}' not located.")
        exit(1)
    
    cv_content = parse_document_content(document_path)

    if not cv_content:
        print("Error: Unable to parse document content.")
        exit(1)

    print("\n===== Parsed CV Content =====\n")
    print(cv_content)

    # Optional position requirements
    position_desc = input("\nProvide position requirements (optional, press Enter to skip): ").strip()
    position_desc = position_desc if position_desc else None

    print("\n===== CV Evaluation Results =====\n")
    print(evaluate_cv_content(cv_content, position_desc))
