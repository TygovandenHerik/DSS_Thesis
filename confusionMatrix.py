#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("/Users/tygovandenherik/Documents/Thesis/Code/data/predictions.csv")

# --- canonicalise once ------------------------------------
def canon(x: str) -> str:
    x = str(x).strip().lower()
    return "PHISHING" if x in ("phish", "phishing", "spam", "1") else "LEGITIMATE"

df["label_true"] = df.label_true.apply(canon)
df["label_pred"] = df.pred.apply(canon)

# pick best models (adjust if needed)
BEST_SCREEN = ("claude35", "screenshot")
BEST_TEXT   = ("claude35", "ocr")

def plot_cm(model, variant, tag):
    sub = df[(df.model == model) & (df.variant == variant)]
    cm  = confusion_matrix(sub.label_true, sub.pred,
                           labels=["PHISHING", "LEGITIMATE"])
    disp = ConfusionMatrixDisplay(cm,
                                  display_labels=["Phish", "Legit"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{model} – {variant}")
    plt.tight_layout()
    fname = f"/Users/tygovandenherik/Documents/Thesis/Code/fig/confmat_{tag}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved", fname)

plot_cm(*BEST_SCREEN, tag="screen")
plot_cm(*BEST_TEXT,   tag="text")
