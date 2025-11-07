import os
import re
import pickle
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer

# ----------------------------
# Configuration
# ----------------------------
# Data
DATA_PATH = "Resume.csv"

# Column name candidates for input text and labels
TEXT_COLUMNS_CANDIDATES: List[str] = ["Resume_str", "Resume", "Resume_text", "ResumeText", "Text"]
LABEL_COLUMNS_CANDIDATES: List[str] = ["Category", "Label"]

# Model & features
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TFIDF_NGRAM_RANGE = (1, 2)       # kept from original approach if you want to switch later
TFIDF_MAX_FEATURES = 15000       # kept for consistency with your prior setup

# Training
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_WEIGHT = "balanced"
MAX_ITER = 20000

# Output
MODEL_OUTPUT_PATH = "resume_classifier.pkl"
EMBEDDING_OUTPUT_PATH = "resume_embedding_model"
LOG_OUTPUT_PATH = None  # set to "training_config.txt" to enable logging

# ----------------------------
# Helpers
# ----------------------------
def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first matching column name from candidates (case-insensitive)."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, remove non-alphanumeric except common symbols, collapse spaces."""
    t = str(text)
    t = t.lower()
    t = re.sub(r"[^a-z0-9+#.\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ----------------------------
# Data loading & validation
# ----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("❌ Loaded dataset is empty.")
    return df

def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    resume_col = find_column(list(df.columns), TEXT_COLUMNS_CANDIDATES)
    if resume_col is None:
        raise ValueError("❌ Could not detect a text column for resumes. Provide one of: " + ", ".join(TEXT_COLUMNS_CANDIDATES))
    label_col = find_column(list(df.columns), LABEL_COLUMNS_CANDIDATES)
    if label_col is None:
        raise ValueError("❌ Could not detect a label/target column. Provide one of: " + ", ".join(LABEL_COLUMNS_CANDIDATES))
    return resume_col, label_col

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    print("📄 Loading dataset...")
    df = load_dataset(DATA_PATH)

    resume_col, label_col = detect_columns(df)
    print(f"Detected text column: {resume_col}, label column: {label_col}")

    # Clean text
    print("♻️ Cleaning text...")
    df["Cleaned"] = df[resume_col].apply(clean_text)

    # Features and labels
    X = df["Cleaned"].fillna("").tolist()
    y = df[label_col].astype(str)

    # Embedding model
    print(f"🔠 Loading Embedding Model ({EMBEDDING_MODEL_NAME})...")
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("🧠 Generating Embeddings...")
    X_vec = encoder.encode(X, show_progress_bar=True, convert_to_numpy=True)

    # Train/test split
    print(f"🔀 Splitting dataset (test_size={TEST_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Classifier
    print(f"🤖 Training Classifier (weight={CLASS_WEIGHT})...")
    model = LinearSVC(class_weight=CLASS_WEIGHT, max_iter=MAX_ITER)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Final Model Accuracy: {accuracy * 100:.2f}%")

    # Save model
    print("💾 Saving Classifier...")
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save embeddings model
    encoder.save(EMBEDDING_OUTPUT_PATH)
    print(f"📦 Saved: {MODEL_OUTPUT_PATH} + {EMBEDDING_OUTPUT_PATH}")

    # Optional: log configuration for reproducibility
    if LOG_OUTPUT_PATH:
        with open(LOG_OUTPUT_PATH, "w") as logf:
            logf.write("Training configuration\n")
            logf.write(f"DATA_PATH={DATA_PATH}\n")
            logf.write(f"EMBEDDING_MODEL_NAME={EMBEDDING_MODEL_NAME}\n")
            logf.write(f"TFIDF_NGRAM_RANGE={TFIDF_NGRAM_RANGE}\n")
            logf.write(f"TFIDF_MAX_FEATURES={TFIDF_MAX_FEATURES}\n")
            logf.write(f"TEST_SIZE={TEST_SIZE}\n")
            logf.write(f"RANDOM_STATE={RANDOM_STATE}\n")
            logf.write(f"CLASS_WEIGHT={CLASS_WEIGHT}\n")

    print("🎉 Training Complete!")

if __name__ == "__main__":
    main()
