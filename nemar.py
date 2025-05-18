#!/usr/bin/env python3
# mcnemar_fixed.py  –  canonicalise labels first

import pandas as pd
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar

def canon(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("phish", "phishing", "spam", "1"):
        return "PHISHING"
    return "LEGITIMATE"          # covers legit / ham / 0

df = pd.read_csv("/Users/tygovandenherik/Documents/Thesis/Code/data/predictions.csv")

# Canonicalise both columns once
df["label_true"]  = df.label_true.apply(canon)
df["label_pred"]  = df.pred.apply(canon)

rows = []
for variant in ("screenshot", "ocr"):
    wide = (df[df.variant == variant]
              .assign(correct=lambda x: x.pred == x.label_true)
              .pivot(index="id", columns="model", values="correct"))

    for m1, m2 in combinations(wide.columns, 2):
        both = wide[[m1, m2]].fillna(False)
        tbl  = [
            [( both[m1] &  both[m2]).sum(), ( both[m1] & ~both[m2]).sum()],
            [(~both[m1] &  both[m2]).sum(), (~both[m1] & ~both[m2]).sum()]
        ]
        stat = mcnemar(tbl, correction=True)
        rows.append(dict(variant=variant,
                         pair=f"{m1} vs {m2}",
                         chi2=round(stat.statistic, 3),
                         p=round(stat.pvalue, 3)))

pd.DataFrame(rows).to_csv("/Users/tygovandenherik/Documents/Thesis/Code/data/mcnemar_pvalues.csv", index=False)
print(pd.DataFrame(rows))
