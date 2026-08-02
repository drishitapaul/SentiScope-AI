import matplotlib
matplotlib.use("Agg")

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

from flask import Flask, render_template, request, send_file

app = Flask(__name__)

# Base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Store report statistics
report_data = {}

# Load trained model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl"))


# ==========================
# Home
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================
# Single Review Prediction
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    probability = model.predict_proba(review_vector)[0]

    confidence = round(max(probability) * 100, 2)

    sentiment = "😊 Positive" if prediction == 1 else "😞 Negative"

    return render_template(
        "index.html",
        review=review,
        prediction=sentiment,
        confidence=confidence
    )


# ==========================
# CSV Upload
# ==========================

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    df = pd.read_csv(file)

    if "review" not in df.columns:
        return "CSV must contain a column named 'review'"

    reviews = df["review"].astype(str)

    vectors = vectorizer.transform(reviews)

    predictions = model.predict(vectors)

    probabilities = model.predict_proba(vectors)

    sentiments = []
    confidence = []

    for pred, prob in zip(predictions, probabilities):

        sentiments.append(
            "Positive" if pred == 1 else "Negative"
        )

        confidence.append(
            round(max(prob) * 100, 2)
        )

    df["Prediction"] = sentiments
    df["Confidence"] = confidence

    output_file = "predictions.csv"

    df.to_csv(output_file, index=False)

    # Dashboard Statistics

    positive_count = sentiments.count("Positive")
    negative_count = sentiments.count("Negative")
    plt.figure(figsize=(5,5))

    plt.pie(
        [positive_count, negative_count],
        labels=["Positive", "Negative"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#6d8b5b", "#d06b8d"]
    )

    plt.title("Sentiment Distribution")

    chart_path = os.path.join(BASE_DIR, "static", "images", "chart.png")
    plt.savefig(chart_path)

    plt.close()

    # ------------------------
    # Dashboard Statistics
    # ------------------------

    avg_confidence = round(
        sum(confidence) / len(confidence), 2
    )

    global report_data

    report_data = {
        "total": len(df),
        "positive": positive_count,
        "negative": negative_count,
        "average": avg_confidence
    }

    return render_template(
        "dashboard.html",
        total=len(df),
        positive=positive_count,
        negative=negative_count,
        average=avg_confidence
    )


# ==========================
# Download CSV
# ==========================

@app.route("/download")
def download():

    return send_file(
        "predictions.csv",
        as_attachment=True
    )                            
# ==========================
# Download PDF Report
# ==========================

@app.route("/download-report")
def download_report():

    pdf = SimpleDocTemplate("SentiScope_Report.pdf")

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph("SentiScope AI Report", styles["Title"])
    )

    story.append(
        Paragraph("<br/>", styles["Normal"])
    )

    story.append(
        Paragraph(
            f"Total Reviews: {report_data['total']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Positive Reviews: {report_data['positive']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Negative Reviews: {report_data['negative']}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"Average Confidence: {report_data['average']}%",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            "Model Accuracy: 89.2%",
            styles["Normal"]
        )
    )

    pdf.build(story)

    return send_file(
        "SentiScope_Report.pdf",
        as_attachment=True
    )


# ==========================
# Run Flask
# ==========================

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)