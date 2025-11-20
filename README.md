# AI-Powered CV Analyzer + ATS Score

An intelligent resume analysis system powered by Machine Learning and Google Gemini AI that helps job seekers optimize their resumes and calculate ATS compatibility scores.

## Features

- **ML-Based Job Classification**: Trained on 2,484 real resumes across 24 job categories
- **Semantic ATS Scoring**: Uses sentence embeddings for intelligent resume-job matching
- **AI-Powered Evaluation**: Google Gemini AI provides comprehensive feedback
- **PDF Text Extraction**: Supports both text-based and scanned PDFs with OCR
- **Web Interface**: Clean, user-friendly Streamlit application

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/resume-analyzer.git
cd resume-analyzer
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Mac**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

### 5. Configure API Key
```bash
cp .env.example .env
```
Edit `.env` and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### 6. Train the Model
```bash
python train_model.py
```
Note: Requires `Resume.csv` dataset (not included - use your own dataset)

## Usage

### Run the Web Application
```bash
streamlit run app.py
```

### How to Use
1. Upload your resume (PDF format)
2. Paste the job description
3. Click "Get AI Evaluation" for detailed feedback
4. Click "Calculate ATS Score" for compatibility score

## How It Works

### ML-Based ATS Scoring
- **Semantic Similarity (70%)**: Compares resume and job description using sentence embeddings
- **Category Match (30%)**: Bonus if predicted job categories align
- **Result**: Accurate 0-100% compatibility score

### Job Classification
- Trained LinearSVC model on 2,484 resumes
- 24 job categories (IT, Finance, Healthcare, etc.)
- Uses SentenceTransformer embeddings (all-MiniLM-L6-v2)

## Project Structure
```
resume_analyzer/
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
└── README.md                  # Documentation
```

## Requirements

- Python 3.7+
- Tesseract OCR
- Google Gemini API key
- Resume dataset (for training)

## Technologies Used

- **ML Framework**: scikit-learn (LinearSVC)
- **Embeddings**: SentenceTransformers
- **AI**: Google Gemini 2.0 Flash
- **Web Framework**: Streamlit
- **PDF Processing**: pdfplumber, pdf2image, pytesseract

## Dataset

The model is trained on a resume dataset with:
- 2,484 resumes
- 24 job categories
- Columns: ID, Resume_str, Resume_html, Category

Note: Dataset not included in repository. Use your own or find publicly available resume datasets.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Google Gemini AI for evaluation capabilities
- SentenceTransformers for semantic embeddings
- Streamlit for the web framework

