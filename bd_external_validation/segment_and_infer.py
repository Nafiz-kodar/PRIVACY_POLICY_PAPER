"""
Phase C/D/F of PROJECT_CONTEXT.md §3.4: sentence-segment the scraped BD policy
text, run the checkpointed BERT classifier over each sentence, and compare the
predicted category distribution against OPP-115's ground truth.

Sentence-level segmentation (not clause-level) is used deliberately: Finding B in
PROJECT_CONTEXT.md (§4.5) shows ~34.9% of whole clauses are multi-category, so
classifying at clause granularity would blend categories the model was trained to
keep separate. Preprocessing before BERT reuses the exact "minimal" variant
(HTML-strip, lowercase, whitespace-normalize -- no stopword removal or
lemmatization) that project_440.ipynb trains BERT on, so inference-time text
matches train-time text.

Requires:
  bd_external_validation/data/bd_extracted_text.csv   (from scrape_bd_policies.py)
  models/bert_policy_checkpoint/                       (from train_bert_checkpoint.py)

Writes:
  bd_external_validation/data/bd_sentence_predictions.csv
  bd_external_validation/data/bd_vs_opp115_category_comparison.csv

Run:
    .venv/bin/python bd_external_validation/segment_and_infer.py
"""
import csv
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging as _logging

_logging.getLogger("transformers").setLevel(_logging.ERROR)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_PATH = os.path.join(REPO_ROOT, "bd_external_validation", "data", "bd_extracted_text.csv")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "models", "bert_policy_checkpoint")
OPP115_PATH = os.path.join(REPO_ROOT, "data", "processed", "privacy_policy_dataset.csv")
PRED_OUT = os.path.join(REPO_ROOT, "bd_external_validation", "data", "bd_sentence_predictions.csv")
COMPARISON_OUT = os.path.join(REPO_ROOT, "bd_external_validation", "data",
                               "bd_vs_opp115_category_comparison.csv")

MIN_WORDS = 4       # drop nav/footer fragments and bare headings
MAX_WORDS = 120      # drop wall-of-text blocks a sentence tokenizer failed to split
MIN_PAGE_CHARS = 500  # requests+BeautifulSoup only sees server-rendered HTML; several BD
                      # sites are JS-rendered SPAs and return a near-empty shell (a handful
                      # of chars) with status "ok" -- treat those as failed extraction too
BATCH_SIZE = 32

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")


def minimal_clean(text):
    """Must match train_bert_checkpoint.py's minimal_clean exactly."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def segment_into_sentences(raw_text):
    """Splits extracted page text into candidate policy sentences, filtering
    obvious boilerplate by word count."""
    sentences = []
    for block in raw_text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for sent in sent_tokenize(block):
            n_words = len(sent.split())
            if MIN_WORDS <= n_words <= MAX_WORDS:
                sentences.append(sent)
    return sentences


def load_checkpoint():
    with open(os.path.join(CHECKPOINT_DIR, "label_map.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_DIR, attn_implementation="eager").to(DEVICE)
    model.eval()
    return model, tokenizer, meta["labels"], meta["max_len"]


def predict_batch(model, tokenizer, texts, labels, max_len):
    enc = tokenizer(texts, padding="max_length", truncation=True,
                     max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=enc["input_ids"].to(DEVICE),
                         attention_mask=enc["attention_mask"].to(DEVICE))
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    return [labels[i] for i in pred_idx], confidence


def main():
    extracted = pd.read_csv(EXTRACTED_PATH)
    ok_rows = extracted[(extracted["status"] == "ok") & (extracted["char_count"] >= MIN_PAGE_CHARS)]
    thin = extracted[(extracted["status"] == "ok") & (extracted["char_count"] < MIN_PAGE_CHARS)]
    print(f"{len(ok_rows)}/{len(extracted)} companies had usable extracted text "
          f"(>= {MIN_PAGE_CHARS} chars)")
    if len(thin):
        print(f"Dropped {len(thin)} as likely JS-rendered / empty shell (< {MIN_PAGE_CHARS} chars): "
              + ", ".join(thin["Company Name"].tolist()))

    model, tokenizer, labels, max_len = load_checkpoint()
    print(f"Loaded checkpoint: {len(labels)} classes, max_len {max_len}, device {DEVICE}")

    records = []
    for _, row in ok_rows.iterrows():
        sentences = segment_into_sentences(str(row["extracted_text"]))
        if not sentences:
            continue
        cleaned = [minimal_clean(s) for s in sentences]

        for start in range(0, len(cleaned), BATCH_SIZE):
            batch_raw = sentences[start:start + BATCH_SIZE]
            batch_clean = cleaned[start:start + BATCH_SIZE]
            preds, confs = predict_batch(model, tokenizer, batch_clean, labels, max_len)
            for sent, pred, conf in zip(batch_raw, preds, confs):
                records.append({
                    "No.": row["No."], "Company Name": row["Company Name"],
                    "Industry": row["Industry"], "text": sent,
                    "predicted_category": pred, "confidence": round(float(conf), 4),
                })

        print(f"[{row['No.']:>3}] {row['Company Name']}: {len(sentences)} sentences classified", flush=True)

    pred_df = pd.DataFrame(records)
    pred_df.to_csv(PRED_OUT, index=False)
    print(f"\nWrote {len(pred_df)} sentence-level predictions -> {PRED_OUT}")

    bd_dist = pred_df["predicted_category"].value_counts(normalize=True).reindex(labels).fillna(0) * 100
    bd_mean_conf = pred_df.groupby("predicted_category")["confidence"].mean().reindex(labels)

    opp115 = pd.read_csv(OPP115_PATH)
    opp_dist = opp115["category"].value_counts(normalize=True).reindex(labels).fillna(0) * 100

    comparison = pd.DataFrame({
        "category": labels,
        "opp115_pct": opp_dist.values.round(2),
        "bd_predicted_pct": bd_dist.values.round(2),
        "gap_pct_points": (bd_dist.values - opp_dist.values).round(2),
        "bd_mean_confidence": bd_mean_conf.values.round(4),
    }).sort_values("gap_pct_points")

    comparison.to_csv(COMPARISON_OUT, index=False)
    print(f"\nCategory distribution: BD (predicted) vs OPP-115 (ground truth)\n")
    print(comparison.to_string(index=False))
    print(f"\nWrote {COMPARISON_OUT}")
    print(f"\nOverall mean prediction confidence on BD text: {pred_df['confidence'].mean():.4f}")


if __name__ == "__main__":
    main()
