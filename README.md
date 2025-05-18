# Multimodal LLMs for Phishing Detection

This repository contains the data, code, and experiment logs used in the master's thesis:

**"Comparing Multimodal Language Models for Phishing Detection: Performance, Cost, and Error Analysis"**  
Tilburg University — MSc Data Science & Society (2025)

## 🧠 Thesis Summary

This study compares the phishing detection capabilities of three state-of-the-art multimodal LLMs:

- **GPT-4o** (OpenAI)  
- **Claude 3.5 Sonnet** (Anthropic)  
- **Gemini 1.5 Pro** (Google)

Each model is tested on 50 real-world e-mails (25 phishing + 25 legitimate), using both the raw screenshot (image prompt) and OCR-extracted text (text prompt). The research investigates:

- Performance (accuracy & macro-F1)
- Latency and cost per model
- Impact of input modality (text vs. image)
- Common error patterns across models

## 📂 Repository Structure

