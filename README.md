# AI CV Assessment Platform

An intelligent CV evaluation system powered by Google Gemini AI that assists professionals in optimizing their CVs and comparing them against position requirements.

## Features

- **PDF Text Extraction**: Supports both text-based and image-based PDFs using OCR
- **AI-Powered Analysis**: Uses Google Gemini AI for comprehensive resume evaluation
- **Job Matching**: Compare resumes against specific job descriptions
- **ATS Compatibility Scoring**: Benchmark resume against job-specific keywords from dataset
- **Skills Assessment**: Identifies existing skills and suggests improvements
- **Course Recommendations**: Suggests relevant courses and learning platforms
- **Web Interface**: User-friendly Streamlit web application

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-analyzer.git
cd resume-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

5. (Optional) Train AI model and generate keyword database:
```bash
# Train job classification model
python train_model.py

# Generate ATS keyword database  
python setup_keywords.py
```
Note: Both require Resume.csv dataset file.

## Getting Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## Usage

### Web Application
```bash
streamlit run app.py
```

### Command Line
```bash
python resume_analyzer.py
```

## Requirements

- Python 3.7+
- Tesseract OCR (for image-based PDFs)
- Google API key for Gemini AI

## Dependencies

- streamlit
- google-generativeai
- pdfplumber
- pdf2image
- pytesseract
- python-dotenv
- Pillow
- pandas
- scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.