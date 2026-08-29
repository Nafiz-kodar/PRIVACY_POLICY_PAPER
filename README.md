# **Leakage-Aware Privacy Practice Classification- A Comparative Study of Classical and Neural Models under Policy-Disjoint Evaluation.**

**A Comparative Study of Classical and Neural Models under Policy-Disjoint Evaluation & Training Leakage-Aware Approach**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Course](https://img.shields.io/badge/CSE440-BRAC%20University-8A2BE2.svg)](#)


---

## Overview

Privacy policies govern how personal data is collected, shared, and retained, yet most users never read them in full. This project studies automatic classification of annotated privacy-policy spans into **eight data-practice categories** using the [OPP-115 corpus](https://usableprivacy.org/data) (17,343 labeled spans from 115 real-world website privacy policies).

The central contribution is a systematic comparison of two evaluation protocols:

- **Row-level stratified split** — the conventional approach, which shuffles individual spans into train/test regardless of which policy they came from.
- **Policy-disjoint split** — an honest evaluation that holds entire policies out of training.

Because privacy policies reuse legally vetted boilerplate language across unrelated organizations, the row-level split lets models score well by memorizing recurring phrases rather than by learning the underlying category, inflating macro-F1 by **0.015–0.123** depending on the model and changing the apparent ranking of the best model entirely. Under honest, policy-disjoint evaluation, **BERT Base achieves the best macro-F1 (0.706)**, narrowly ahead of Naive Bayes (0.701) — a near-tie that also shows the smallest evaluation-protocol gap of any model tested.

Ten models were trained and tuned across three classical approaches, six recurrent architectures, and one transformer, for a combined 60 logged tuning runs.

## Key Results

Test-set performance at each model's best validated configuration, ranked by macro-F1 under policy-disjoint evaluation:

| Model | Random Split Acc. | Random Split Macro-F1 | Policy-Disjoint Acc. | Policy-Disjoint Macro-F1 |
|---|---|---|---|---|
| **BERT Base** | 0.759 | 0.721 | 0.715 | **0.706** |
| Naive Bayes | 0.757 | 0.726 | 0.703 | 0.701 |
| Logistic Regression | 0.774 | 0.753 | 0.679 | 0.692 |
| Random Forest | 0.780 | 0.721 | 0.677 | 0.652 |
| Bidirectional LSTM | 0.760 | 0.684 | 0.670 | 0.608 |
| Bidirectional GRU | 0.764 | 0.700 | 0.633 | 0.586 |
| Bidirectional SimpleRNN | 0.728 | 0.612 | 0.617 | 0.489 |
| GRU | 0.711 | 0.519 | 0.585 | 0.475 |
| LSTM | 0.691 | 0.549 | 0.579 | 0.442 |
| SimpleRNN | 0.482 | 0.172 | 0.386 | 0.131 |

Full methodology, per-class analysis, and discussion are in the [final report](final%20report/Leakage-Aware%20Privacy%20Practice%20Classification-%20A%20Comparative%20Study%20of%20Classical%20and%20Neural%20Models%20under%20Policy-Disjoint%20Evaluation.pdf).

**Takeaway:** row-level evaluation on this corpus is a genuine trap — Random Forest posts the *highest accuracy of all ten models* (0.780) under the leaky split while being the *weakest of the three classical models* on macro-F1 (0.652) once policy-level leakage is removed. Document-grouped evaluation should be standard practice for text classification built on corpora with reused or template-based language.

## Repository Structure

```
Project
├── project_440.ipynb                    notebook: preprocessing, all 10 models, both eval protocols
├── raw/                                     data (OPP-115 + a supplementary BD company registry)
├── processed/                                  engineered datasets (5 preprocessing variants + final dataset)
├── final report/                            paper (LaTeX source, figures, compiled PDF)
├── deploy/                                  Next.js app serving the fine-tuned BERT Base model
└── LICENSE
```

## Dataset

- **Source:** [OPP-115](https://usableprivacy.org/data) (Wilson et al., 2016) — 115 real-world website privacy policies, annotated by legal experts against a fixed data-practice taxonomy.
- **Size:** 17,343 labeled spans across 8 categories: *First Party Collection/Use, Third Party Sharing/Collection, User Choice/Control, Data Security, User Access Edit and Deletion, International and Specific Audiences, Policy Change, Data Retention.*
- **Preprocessing variants:** `processed/` contains five engineered variants (base, minimal, aggressive, lemmatized, stopwords-removed) used to test representation robustness, plus the final assembled dataset.
- **Representation status:** TF-IDF is the only representation with completed downstream classification (used by all classical model results). A Word2Vec pipeline was implemented but not run to completion; GloVe was not implemented.

## Models

| Family | Models |
|---|---|
| Classical | Logistic Regression, Naive Bayes, Random Forest (TF-IDF features) |
| Recurrent | SimpleRNN, GRU, LSTM, and their bidirectional variants (6 total) |
| Transformer | BERT Base (partial, layer-frozen fine-tuning) |

## Live Demo

`deploy/` contains a self-contained Next.js application that serves the fine-tuned BERT Base checkpoint directly via an in-browser/server ONNX runtime (`@huggingface/transformers`) — no external API calls at inference time.

### Running Locally

```bash
cd deploy
npm install
npm run dev
```

Then open **http://localhost:3000** in your browser, paste a privacy-policy sentence, and click **Classify**. The first request loads the model into memory; subsequent requests are fast (~15ms).

### Deployment

This app is ready to deploy to [Vercel](https://vercel.com). The model files are self-contained and Git LFS is configured (`.gitattributes`). After pushing to GitHub, add the repo as a new Vercel project and enable Git LFS in project Settings → Git before deploying.

> This demo is an exploratory research artifact, not a compliance, legal, or safety verdict about any policy or company.

## Getting Started

```bash
git clone https://github.com/Nafiz-kodar/PRIVACY_POLICY_PAPER.git
cd PRIVACY_POLICY_PAPER
git lfs install   # required — this repo tracks large model/report assets via Git LFS
```

Open [project_440.ipynb](project_440.ipynb) in Jupyter or Google Colab to reproduce preprocessing, training, and evaluation for all ten models under both split schemes.

## Report

The full write-up — methodology, per-class error analysis, evaluation-protocol gap analysis, limitations, and future work — is available as:

- LaTeX source: [`final report/report.tex`](final%20report/report.tex)
- Compiled PDF: [`final report/`](final%20report/Leakage-Aware%20Privacy%20Practice%20Classification-%20A%20Comparative%20Study%20of%20Classical%20and%20Neural%20Models%20under%20Policy-Disjoint%20Evaluation.pdf).

## Authors

| Name | Email |
|---|---|
| Nafiz Ahmed Nafi | nafiz.ahmed.nafi@g.bracu.ac.bd |
| Priom Halder | priom.halder@g.bracu.ac.bd |
| MD. Amirul Islam Sadat | amirul.islam.sadat@g.bracu.ac.bd |
| Samiha Tasnim Orthi | samiha.tasnim.orthi@g.bracu.ac.bd |

Department of Computer Science and Engineering, BRAC University

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Built on the [OPP-115 corpus](https://usableprivacy.org/data) (Wilson et al., 2016, *The Creation and Analysis of a Website Privacy Policy Corpus*). Submitted as coursework for CSE440 (Natural Language Processing Learning), BRAC University.
