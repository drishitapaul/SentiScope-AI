"""
02_train_model.py
Train a text classifier: TF-IDF vectorization + Logistic Regression.
This combo is chosen deliberately over a neural network - it's fast to train,
easy to explain in an interview, and gets strong results on this kind of
task, which matters more for a portfolio project than squeezing out the last
1% of accuracy with a heavier model.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

df = pd.read_csv("data/reviews_clean.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# TF-IDF: turns each review into a vector of word-importance scores.
# max_features caps vocabulary size to keep it fast and avoid overfitting on rare words.
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred, target_names=['negative', 'positive'])}")

# Save model + vectorizer so they can be reused without retraining
joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

# Save metrics + confusion matrix for the README/chart script
import json
with open("data/metrics.json", "w") as f:
    json.dump({
        "train_size": len(X_train), "test_size": len(X_test),
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm.tolist()
    }, f, indent=2)

# Show the words the model weighted most heavily for each class - good for
# explaining "what did the model actually learn" in an interview
feature_names = vectorizer.get_feature_names_out()
coefs = model.coef_[0]
top_positive = sorted(zip(coefs, feature_names), reverse=True)[:15]
top_negative = sorted(zip(coefs, feature_names))[:15]

print("\nTop words pushing toward POSITIVE:")
for c, w in top_positive:
    print(f"  {w:20s} {c:.3f}")
print("\nTop words pushing toward NEGATIVE:")
for c, w in top_negative:
    print(f"  {w:20s} {c:.3f}")

pd.DataFrame(top_positive, columns=["weight", "word"]).to_csv("data/top_positive_words.csv", index=False)
pd.DataFrame(top_negative, columns=["weight", "word"]).to_csv("data/top_negative_words.csv", index=False)
