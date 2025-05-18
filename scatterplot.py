import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/tygovandenherik/Documents/Thesis/Code/data/metrics_table.csv")

plt.figure(figsize=(4.5, 3.5))

markers = {"screenshot": "o", "ocr": "s"}
for _, row in df.iterrows():
    plt.scatter(row.euro_per_mail,
                row.accuracy,
                marker=markers[row.variant],
                s=70,
                label=f"{row.model}-{row.variant}")

plt.xlabel("Cost per mail (€)")
plt.ylabel("Accuracy")
plt.xlim(left=0)
plt.ylim(0.8, 1.0)
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("/Users/tygovandenherik/Documents/Thesis/Code/fig/scatter_cost_vs_acc.png", dpi=300)
print("Saved fig/scatter_cost_vs_acc.png")
