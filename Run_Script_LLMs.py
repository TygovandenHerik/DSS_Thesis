#!/usr/bin/env python3
# ------------------------------------------------------------------
# run_experiment.py  –  Calls GPT-4o, Claude-3.5, Gemini-1.5 on each
# email screenshot (vision) and OCR text; logs predictions & tokens.
#
# • Requires  data/emails.csv  with cols:
#     id,label,text,image_path
# • Requires  config.json  with your API keys:
#   {
#     "openai_api_key"   : "sk-...",
#     "anthropic_api_key": "claude-...",
#     "google_api_key"   : "AIza..."
#   }
# ------------------------------------------------------------------
import json, time, base64, argparse
import pandas as pd
from pathlib import Path
import openai, anthropic, google.generativeai as genai
from PIL import Image
import re
import time, google.generativeai as genai
import google.api_core.exceptions as gex

LABEL_RE = re.compile(r"\b(PHISHING|LEGITIMATE)\b", re.I)
def extract_label(text: str) -> str:
    m = LABEL_RE.search(text)
    return m.group(1).upper() if m else "UNKNOWN"

# ---------- CLI arg ----------
argp = argparse.ArgumentParser()
argp.add_argument("--n", type=int, default=None,
                  help="Process only N e-mails (pilot)")
args = argp.parse_args()


# ---------- Load API keys ----------
with open("/Users/tygovandenherik/Documents/Thesis/Code/config.json") as f:
    cfg = json.load(f)
openai_client    = openai.OpenAI(api_key=cfg["openai_api_key"])
anthropic_client = anthropic.Anthropic(api_key=cfg["anthropic_api_key"])
genai.configure(api_key=cfg["google_api_key"])

# ---------- Prompts ----------
SYS_PROMPT = (
    "You are an e-mail security assistant at a SOC. "
    "Reply with exactly PHISHING or LEGITIMATE—no explanations."
)

# vision variant ─ no placeholder needed
SCR_USER = (
    "User: Here is an e-mail screenshot.\n\n"
    "Reply with only one of these words (no punctuation): "
    "PHISHING or LEGITIMATE."
)

# text variant ─ one placeholder {EMAIL}
TXT_USER_T = (
    "User: Here is the full e-mail:\n\n{EMAIL}\n\n"
    "Is this e-mail phishing?\n\n"
    "Reply with only one of these words (no punctuation): "
    "PHISHING or LEGITIMATE."
)

# ---------- Dataset ----------
df = pd.read_csv("/Users/tygovandenherik/Documents/Thesis/Code/data/emails.csv")
if args.n:
    df = df.head(args.n)
print(f"Processing {len(df)} e-mails …")

# ---------- Model wrappers ---------------------------------------
def gpt4o_call(img_path=None, email_text=None):
    msgs = [{"role": "system", "content": SYS_PROMPT}]
    if img_path:
        with open(img_path, "rb") as f:
            data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": SCR_USER},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}}
        ]})
    else:
        msgs.append({"role": "user",
                     "content": TXT_USER_T.format(EMAIL=email_text)})
    t0 = time.time()
    r  = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=msgs,
            temperature=0,
            max_tokens=3)
    lat = (time.time() - t0) * 1000
    u   = r.usage
    return r.choices[0].message.content.strip(), u.prompt_tokens, u.completion_tokens, lat

VALID_RE = re.compile(r"\b(PHISHING|LEGITIMATE)\b", re.I)

def claude_call(img_path=None, email_text=None):
    if img_path:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content = [
            {"type": "image",
             "source": {"type": "base64",
                        "media_type": "image/png",
                        "data": b64}},
            {"type": "text", "text": SCR_USER}
        ]
    else:
        content = [{"type": "text",
                    "text": TXT_USER_T.format(EMAIL=email_text)}]

    t0 = time.time()
    r = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": content}],
            max_tokens=8, temperature=0)
    latency = (time.time() - t0) * 1000
    usage   = r.usage
    raw     = r.content[0].text.strip()
    m       = VALID_RE.search(raw)
    pred    = m.group(1).upper() if m else raw[:15].upper()  # fallback
    return pred, usage.input_tokens, usage.output_tokens, latency

import time, google.generativeai as genai

def gemini_call(img_path=None, email_text=None):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    cfg   = genai.types.GenerationConfig(max_output_tokens=5, temperature=0)

    def _one_call():
        if img_path:            # screenshot
            img = Image.open(img_path)
            return model.generate_content([SCR_USER, img],
                                          generation_config=cfg)
        else:                   # OCR text
            return model.generate_content(
                TXT_USER_T.format(EMAIL=email_text),
                generation_config=cfg)

    t0 = time.time()
    try:
        resp = _one_call()
    except gex.ResourceExhausted:
        print("Gemini free-tier quota hit → waiting 65 s …")
        time.sleep(65)
        resp = _one_call()

    latency = (time.time() - t0) * 1000
    usage   = resp.usage_metadata
    tok_total = getattr(usage, "total_token_count",
                getattr(usage, "tokenCount", 0))   # ← new primary field


    pred = resp.text.strip().split()[0].upper()   # first word
    return pred, tok_total, 0, latency

CALL = {"gpt4o": gpt4o_call,
        "claude35": claude_call,
        "gemini15": gemini_call}

# ---------- Main loop --------------------------------------------
if __name__ == '__main__':   
    records = []
    for _, row in df.iterrows():
        for variant in ("screenshot", "ocr"):
            img = row.image_path if variant == "screenshot" else None
            txt = row.text       if variant == "ocr"        else None
            for mkey, func in CALL.items():
                pred, tin, tout, lat = func(img, txt)
                records.append(dict(id=row.id, label_true=row.label,
                                    variant=variant, model=mkey,
                                    pred=pred, tok_in=tin, tok_out=tout, lat_ms=lat))
                print(f"{row.id:<12} {variant:<10} {mkey:<8} → {pred}")
                
    out_csv = Path("/Users/tygovandenherik/Documents/Thesis/Code/data/predictions.csv")
    out_csv.parent.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False, encoding="utf-8")
    print("Saved predictions to", out_csv)
