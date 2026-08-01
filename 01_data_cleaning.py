"""
01_data_cleaning.py
Business question: Can we automatically tell whether a customer review is
positive or negative, accurately enough to be useful (e.g. for flagging
negative reviews for follow-up, or tracking sentiment trends over time)?

Data: 50,000 IMDB movie reviews, labeled positive/negative (originally
compiled by Maas et al., a standard benchmark dataset for this exact task).

Cleaning steps:
  - Remove duplicate reviews
  - Strip HTML tags (the raw text has <br /><br /> line breaks in it)
  - Lowercase + strip extra whitespace
"""
import pandas as pd
import re

df = pd.read_csv("data/imdb_reviews.csv")
n_raw = len(df)

df = df.drop_duplicates(subset="review")
n_dupes = n_raw - len(df)

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)          # strip HTML tags like <br />
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # keep only letters
    text = re.sub(r"\s+", " ", text).strip()     # collapse whitespace
    return text.lower()

df["clean_review"] = df["review"].apply(clean_text)
df["label"] = (df["sentiment"] == "positive").astype(int)  # 1 = positive, 0 = negative

df[["clean_review", "sentiment", "label"]].to_csv("data/reviews_clean.csv", index=False)

print(f"Raw rows:          {n_raw:,}")
print(f"Duplicates removed: {n_dupes:,}")
print(f"Clean rows:         {len(df):,}")
print(f"Class balance:\n{df['sentiment'].value_counts()}")
print(f"\nExample before cleaning:\n{df['review'].iloc[0][:200]}")
print(f"\nExample after cleaning:\n{df['clean_review'].iloc[0][:200]}")
