import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("📄 Loading dataset...")
df = pd.read_csv("Resume.csv")

def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    return text

# Detect resume column
if "Resume" in df.columns:
    resume_col = "Resume"
elif "Resume_str" in df.columns:
    resume_col = "Resume_str"
else:
    print("❌ ERROR: Resume column not found! Available columns:")
    print(df.columns)
    exit()

print(f"✅ Using resume text column: {resume_col}")

print("🧹 Cleaning text...")
df["Cleaned"] = df[resume_col].apply(clean_text)

X = df["Cleaned"]
y = df["Category"]

print("🔠 Converting text to TF-IDF vectors...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

print("🔀 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("📊 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

print("💾 Saving model and vectorizer...")
with open("resume_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("resume_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("🎉 Done! Model Saved Successfully!")
