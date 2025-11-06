# Dataset Setup Instructions

## Step 1: Get Resume Dataset

Download a resume dataset CSV file with these columns:
- `Resume` - Contains resume text
- `Category` - Job category/role (e.g., "Data Science", "Web Development")

**Popular sources:**
- Kaggle: Search for "resume dataset" or "job classification dataset"
- GitHub: Look for resume classification datasets

## Step 2: Prepare Dataset

1. Save the dataset as `Resume.csv` in your project folder
2. Ensure it has the required columns: `Resume` and `Category`

## Step 3: Generate Keyword Database

Run the setup script:
```bash
python setup_keywords.py
```

This will:
- Read Resume.csv
- Extract top keywords for each job category
- Save as `position_keywords.pkl`

## Step 4: Verify Setup

The script will show:
```
📊 Loading Resume.csv...
🔍 Processing 25 job categories...
  ✅ Data Science: 50 keywords
  ✅ Web Development: 50 keywords
  ✅ HR: 50 keywords
  ...
🎉 Successfully created keyword database!
```

## Step 5: Use in App

Once `position_keywords.pkl` exists, the Streamlit app will automatically show ATS scoring options.

## Sample Dataset Format

```csv
Resume,Category
"Python developer with 3 years experience in Django Flask...","Web Development"
"Data scientist skilled in machine learning pandas numpy...","Data Science"
"HR professional with recruiting and training experience...","HR"
```