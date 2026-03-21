# train_model.py
# 🚀 Spam Detection Model Training

import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

# ------------------------
# 1️⃣ LOAD DATASET
# ------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1","v2"]]  # Keep only relevant columns
df.columns = ["label","message"]

# ------------------------
# 2️⃣ CONVERT LABELS
# ham → 0, spam → 1
# ------------------------
df["label"] = df["label"].map({"ham":0,"spam":1})

# ------------------------
# 3️⃣ CLEAN TEXT FUNCTION
# ------------------------
def clean_text(text):
    text = text.lower()                       # Lowercase all letters
    text = re.sub(r'[^a-z]', ' ', text)      # Remove everything except letters
    return text

df["message"] = df["message"].apply(clean_text)

# ------------------------
# 4️⃣ FEATURE EXTRACTION
# ------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2))  # Unigrams + Bigrams
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# ------------------------
# 5️⃣ SPLIT DATA
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# 6️⃣ TRAIN MODEL
# ------------------------
model = LogisticRegression(max_iter=1000)  # Increase iterations to avoid warnings
model.fit(X_train, y_train)

# ------------------------
# 7️⃣ PREDICT & EVALUATE
# ------------------------
pred = model.predict(X_test)
print("Accuracy  :", accuracy_score(y_test, pred))
print("Precision :", precision_score(y_test, pred))
print("Recall    :", recall_score(y_test, pred))
cm = confusion_matrix(y_test, pred)
print(cm)

# ------------------------
# 8️⃣ SAVE MODEL & VECTORIZER
# ------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")
