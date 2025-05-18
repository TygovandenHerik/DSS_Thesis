import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PRICE = {
    "gpt4o":    {"in": 0.005,   "out": 0.020},
    "claude35":  {"in": 0.003,   "out": 0.015},
    "gemini15":  {"in": 0.00125, "out": 0.005}
}

df = pd.read_csv("/Users/tygovandenherik/Documents/Thesis/Code/data/predictions.csv")
rows = []


def canon(s: str) -> str:
    """Map any label form to canonical 'PHISHING' or 'LEGITIMATE'."""
    s = str(s).strip().lower()
    if s in ("phish", "phishing", "spam", "1"):
        return "PHISHING"
    elif s in ("legit", "legitimate", "ham", "0"):
        return "LEGITIMATE"
    else:
        raise ValueError(f"Unknown label: {s}")

rows = []
for (model, variant), grp in df.groupby(["model", "variant"]):
    y_true = grp.label_true.apply(canon)
    y_pred = grp.pred.apply(canon)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro")
    lat  = grp.lat_ms.mean()
    cost = ((grp.tok_in  / 1000) * PRICE[model]["in"] +
            (grp.tok_out / 1000) * PRICE[model]["out"]).mean()

    rows.append(dict(model=model,
                     variant=variant,
                     accuracy=acc,
                     f1=f1,
                     latency_ms=lat,
                     euro_per_mail=cost))


out = pd.DataFrame(rows)
out.to_csv("/Users/tygovandenherik/Documents/Thesis/Code/data/metrics_table.csv", index=False)
print(out.round(3))
