from flask import Flask, request, render_template
import pickle
from textblob import TextBlob
import re

# ------------------------
# 1️⃣ CLEAN TEXT + SPELL CORRECTION
# ------------------------
def clean_text(text):
    # Spell correction
    corrected = str(TextBlob(text).correct())
    # Lowercase and remove non-letters
    corrected = corrected.lower()
    corrected = re.sub(r'[^a-z ]', '', corrected)
    return corrected

# ------------------------
# 2️⃣ LOAD MODEL & VECTORIZER
# ------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ------------------------
# 3️⃣ SAMPLE SPAM WORDS LIST
# ------------------------
spam_words = ["free", "win", "winner", "cash", "prize", "congratulations", "lottery", "offer", "click"]

# ------------------------
# 4️⃣ CREATE FLASK APP
# ------------------------
app = Flask(__name__)

# ------------------------
# 5️⃣ HOME ROUTE
# ------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    prob_spam = 0
    prob_ham = 0
    highlighted_msg = ""
    threshold = 0.2  # default threshold
    if request.method == "POST":
        msg = request.form["message"]
        threshold = float(request.form.get("threshold", 0.2))
        msg_clean = clean_text(msg)
        msg_vec = vectorizer.transform([msg_clean])
        prob = model.predict_proba(msg_vec)[0]
        prob_ham = prob[0]
        prob_spam = prob[1]
        # Determine result using threshold
        result = "Spam ❌" if prob_spam > threshold else "Not Spam ✅"
        # Highlight spam words
        words = msg.split()
        highlighted_words = []
        for w in words:
            if w.lower() in spam_words:
                highlighted_words.append(f"<span style='color:red'>{w}</span>")
            else:
                highlighted_words.append(w)
        highlighted_msg = " ".join(highlighted_words)

    return render_template(
        "index.html",
        result=result,
        prob_ham=f"{prob_ham*100:.2f}%",
        prob_spam=f"{prob_spam*100:.2f}%",
        highlighted_msg=highlighted_msg,
        threshold=threshold
    )

# ------------------------
# 6️⃣ RUN APP
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)