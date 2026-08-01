"""
03_predict.py
Loads the trained model and lets you test it on any text you type in -
useful for demoing the project live in an interview.
"""
import joblib
import re
import sys

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = prob[pred]
    return label, confidence

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, conf = predict(text)
        print(f"Text: {text}")
        print(f"Prediction: {label} (confidence: {conf:.1%})")
    else:
        # Demo examples
        examples = [
            "This movie was absolutely fantastic, I loved every minute of it!",
            "Waste of two hours. Terrible acting and a boring plot.",
            "It was okay, not great but not terrible either.",
            "One of the best films I've seen this year, brilliant performances.",
        ]
        for ex in examples:
            label, conf = predict(ex)
            print(f"'{ex}'\n  -> {label} (confidence: {conf:.1%})\n")
