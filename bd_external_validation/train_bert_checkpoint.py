"""
Bootstraps a persisted BERT checkpoint for the CSE440 privacy-policy classifier.

project_440.ipynb (cells 50/59/83) trains BERT Base for both the "random" and
"policy" split schemes entirely in-memory -- nothing is ever written to disk, so
the model is lost the moment the kernel restarts. This script replicates only the
"policy" scheme's data prep + training path (the scheme documented in
PROJECT_CONTEXT.md as the honest, reported result: macro-F1 0.6914) and adds the
one missing step: `save_pretrained` after training, so the BD external-validation
pipeline has a real checkpoint to load. It intentionally skips the "random" scheme
and skips retraining every ML/NN model in the notebook -- neither is needed to
produce this checkpoint.

Run:
    .venv/bin/python bd_external_validation/train_bert_checkpoint.py
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging as _logging

_logging.getLogger("transformers").setLevel(_logging.ERROR)
_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(REPO_ROOT, "data", "processed", "privacy_policy_dataset.csv")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "models", "bert_policy_checkpoint")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

CATEGORIES = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "Data Security",
    "User Access, Edit and Deletion",
    "International and Specific Audiences",
    "Policy Change",
    "Data Retention",
]

BERT_MODEL_NAME = "bert-base-uncased"
BERT_MAX_LEN = 64
BERT_MAX_EPOCHS = 3
BERT_PATIENCE = 1
BERT_FROZEN_LAYERS = 10  # of 12 encoder layers -- matches project_440.ipynb cell 50

BERT_GRID = [
    {"lr": 2e-5, "batch_size": 16},
    {"lr": 3e-5, "batch_size": 32},
    {"lr": 5e-5, "batch_size": 16, "class_weight": "balanced"},
]

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")


def minimal_clean(text):
    """Mirrors TextCleaner._process_minimal in project_440.ipynb cell 30 --
    the exact variant BERT is trained on (BERT_PREPROCESS_VARIANT = "minimal")."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def build_policy_scheme(df, val_fraction=0.1667, seed=RANDOM_STATE):
    """Verbatim port of project_440.ipynb cell 38's build_policy_scheme."""
    train_pool = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    dominant = train_pool.groupby("policy_id")["category"].agg(lambda s: s.value_counts().idxmax())
    stratify = dominant.values if (dominant.value_counts() < 2).sum() == 0 else None

    train_pol, val_pol = train_test_split(
        dominant.index.tolist(), test_size=val_fraction,
        stratify=stratify, random_state=seed)

    return {
        "train": train_pool[train_pool["policy_id"].isin(train_pol)].copy(),
        "val": train_pool[train_pool["policy_id"].isin(val_pol)].copy(),
        "test": test.copy(),
    }


def build_bert_encodings(parts, tokenizer, max_len=BERT_MAX_LEN):
    enc = {}
    for key in ("train", "val", "test"):
        tok = tokenizer(
            parts[key]["text_clean_bert"].fillna("").tolist(),
            padding="max_length", truncation=True, max_length=max_len,
            return_tensors="pt")
        enc[key] = {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}
    return enc


def make_bert_loader(enc, y, batch_size, shuffle=False):
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_bert_model(model, loader):
    model.eval()
    predictions, actual = [], []
    with torch.no_grad():
        for input_ids, attention_mask, y in loader:
            outputs = model(input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE))
            pred = torch.argmax(outputs.logits, dim=1)
            predictions.extend(pred.cpu().numpy())
            actual.extend(y.numpy())
    return {
        "accuracy": accuracy_score(actual, predictions),
        "macro_f1": f1_score(actual, predictions, average="macro"),
    }


def train_one_bert_config(params, n_classes, class_counts, labels, enc, y,
                           max_epochs=BERT_MAX_EPOCHS):
    batch_size = params["batch_size"]
    train_loader = make_bert_loader(enc["train"], y["train"], batch_size, shuffle=True)
    val_loader = make_bert_loader(enc["val"], y["val"], batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=n_classes, attn_implementation="eager"
    ).to(DEVICE)

    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:BERT_FROZEN_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=params["lr"])

    if params.get("class_weight") == "balanced":
        freq = np.array([class_counts.get(labels[i], 1) for i in range(n_classes)], dtype=np.float32)
        weights = torch.tensor(freq.sum() / (n_classes * freq), dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_f1, best_state, patience_left = -1.0, None, BERT_PATIENCE
    for epoch in range(1, max_epochs + 1):
        model.train()
        for input_ids, attention_mask, yb in train_loader:
            input_ids, attention_mask, yb = (input_ids.to(DEVICE), attention_mask.to(DEVICE), yb.to(DEVICE))
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, yb)
            loss.backward()
            optimizer.step()

        val = evaluate_bert_model(model, val_loader)
        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1 = val["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = BERT_PATIENCE
        else:
            patience_left -= 1

        print(f"      epoch {epoch:2d}  val macro-F1 {val['macro_f1']:.4f}  "
              f"acc {val['accuracy']:.4f}{'  *' if improved else ''}", flush=True)
        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    return model, best_f1


def save_checkpoint(model, tokenizer, labels):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    with open(os.path.join(CHECKPOINT_DIR, "label_map.json"), "w", encoding="utf-8") as fh:
        json.dump({"labels": labels, "max_len": BERT_MAX_LEN, "model_name": BERT_MODEL_NAME}, fh, indent=2)


def main():
    print(f"Device: {DEVICE}")
    df = pd.read_csv(DATASET_PATH)
    parts = build_policy_scheme(df)

    for key in ("train", "val", "test"):
        parts[key]["text_clean_bert"] = [minimal_clean(t) for t in parts[key]["text"].fillna("").tolist()]

    labels = sorted(parts["train"]["category"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    n_classes = len(labels)
    class_counts = parts["train"]["category"].value_counts().to_dict()

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    enc = build_bert_encodings(parts, tokenizer, max_len=BERT_MAX_LEN)
    y = {split: parts[split]["category"].map(label2id).values.astype(np.int64)
         for split in ("train", "val", "test")}

    print(f"policy scheme -- BERT Base, max_len {BERT_MAX_LEN}, "
          f"train {len(parts['train'])} / val {len(parts['val'])} / test {len(parts['test'])}")

    best_score, best_model, best_params = -1, None, None
    for i, params in enumerate(BERT_GRID, 1):
        print(f"\nConfiguration {i}: {params}", flush=True)
        start = time.time()
        model, score = train_one_bert_config(params, n_classes, class_counts, labels, enc, y)
        print(f"Validation Macro F1: {score:.4f}  ({time.time() - start:.1f}s)", flush=True)

        if score > best_score:
            best_score, best_model, best_params = score, model, params
            save_checkpoint(best_model, tokenizer, labels)
            print(f"  -> new best, checkpoint saved to {CHECKPOINT_DIR}", flush=True)

    test_loader = make_bert_loader(enc["test"], y["test"], best_params["batch_size"])
    test_result = evaluate_bert_model(best_model, test_loader)

    print(f"\nBest config: {best_params}")
    print(f"Validation Macro F1: {best_score:.4f}")
    print(f"Test Macro F1      : {test_result['macro_f1']:.4f}")
    print(f"Test Accuracy      : {test_result['accuracy']:.4f}")
    print(f"\nCheckpoint at {CHECKPOINT_DIR} (already saved after the winning config).")


if __name__ == "__main__":
    main()
