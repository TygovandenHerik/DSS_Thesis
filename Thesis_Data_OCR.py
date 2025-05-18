
from pathlib import Path
import pytesseract, pandas as pd, re
from PIL import Image

IMG_DIR   = Path("/Users/tygovandenherik/Documents/Thesis/Screenshots")        # adjust if your images live elsewhere
OUT_CSV   = Path("/Users/tygovandenherik/Documents/Thesis/emails.csv")
LANG      = "eng"                           # Tesseract language pack to use

rows = []
for img_path in sorted(IMG_DIR.glob("*.png")):
    # OCR
    text = pytesseract.image_to_string(Image.open(img_path), lang=LANG)
    text = text.replace("\u201c", '"').replace("\u201d", '"').strip()  # tidy quotes/whitespace

    # Label from filename: fake_XXX or phish_XXX ⇒ phishing, else legitimate
    if re.match(r"(fake|phish)", img_path.stem, flags=re.I):
        label = "phish"
    else:
        label = "ham"

    rows.append({
        "id":           img_path.stem,
        "label":        label,
        "text":         text,
        "image_path":   str(img_path)
    })

# Save
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

print(f"Saved {len(rows)} rows to {OUT_CSV}")
print(pd.read_csv(OUT_CSV).head(3).to_markdown())
