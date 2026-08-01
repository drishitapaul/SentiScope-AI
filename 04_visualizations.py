import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "figure.dpi": 130})

with open("data/metrics.json") as f:
    m = json.load(f)

# 1. Confusion matrix heatmap
cm = np.array(m["confusion_matrix"])
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Negative", "Positive"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Negative", "Positive"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (Accuracy: {m['accuracy']:.1%})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=13)
plt.tight_layout()
plt.savefig("charts/confusion_matrix.png")
plt.close()

# 2. Metrics bar chart
fig, ax = plt.subplots(figsize=(6, 4))
metrics = ["accuracy", "precision", "recall", "f1"]
values = [m[k] for k in metrics]
bars = ax.bar([k.capitalize() for k in metrics], values, color="#2E5A88")
ax.set_ylim(0, 1)
ax.set_title("Model Performance on Held-Out Test Set")
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.1%}", ha="center")
plt.tight_layout()
plt.savefig("charts/model_metrics.png")
plt.close()

# 3. Top words driving each class
pos = pd.read_csv("data/top_positive_words.csv").head(10)
neg = pd.read_csv("data/top_negative_words.csv").head(10)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].barh(pos["word"][::-1], pos["weight"][::-1], color="#2E8B57")
axes[0].set_title("Top words -> POSITIVE")
axes[1].barh(neg["word"][::-1], neg["weight"][::-1], color="#C0392B")
axes[1].set_title("Top words -> NEGATIVE")
plt.tight_layout()
plt.savefig("charts/top_words.png")
plt.close()

print("Saved 3 charts to charts/")
