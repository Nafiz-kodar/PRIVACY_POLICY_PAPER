# Privacy Policy Clause Classifier — Vercel deployment

CSE440 §1.6 bonus deliverable: a live demo of the project's best-performing model
(**BERT Base**). This is a self-contained Next.js app — it does not import anything
from `code/` at request time. It serves inference itself, in Node.js, via a locally
bundled ONNX export of the fine-tuned weights (`@huggingface/transformers`), with no
external API calls and no Hugging Face Hub access at runtime.

The fine-tuned model is already trained, exported, and committed in this folder
(`models/bert-privacy-policy/`) — this repo is ready to push and deploy as-is.

**Measured performance of the deployed checkpoint** (policy-disjoint split, single
retraining run of the winning configuration from `PROJECT_CONTEXT.md` §8.1 —
batch=16, lr=2e-5): test accuracy **0.7159**, macro-F1 **0.6987**. This is a fresh
training run, not the exact 0.7116/0.6914 numbers in `PROJECT_CONTEXT.md` §8.1 (BERT
fine-tuning is stochastic — dropout, data-loader shuffling), but lands within the
same range and confirms the same conclusion: BERT Base narrowly leads the classical
ceiling under honest, policy-disjoint evaluation. Full metrics, including per-class
F1, are committed in `eval_metrics.json`.

## This repo is self-contained — this is its own git repository

This folder (`deploy/`) is initialized as its **own separate git repository**,
independent of the rest of the CSE440 project (which has no git repo of its own).
That's deliberate: this app doesn't depend on `code/`, `data/`, or anything else in
the project at request time, and keeping it separate means you push only ~110 MB of
model weights to GitHub, not the ~2 GB OPP-115 corpus or training artifacts.

## Git LFS — required

`models/bert-privacy-policy/onnx/model_quantized.onnx` is ~110 MB, over GitHub's
100 MB per-file limit for regular git pushes, so it's tracked via **Git LFS**
(`.gitattributes` already configures this). To push:

```bash
# one-time, if you don't already have git-lfs:
# https://git-lfs.com  (or: winget install GitHub.GitLFS / brew install git-lfs)
git lfs install
```

That's it — `git push` will transparently upload the LFS-tracked file. Vercel pulls
LFS objects automatically when it clones the repo to build.

**GitHub's free LFS bandwidth quota is 1 GB/month**, and every Vercel build that
re-clones the repo pulls this ~110 MB file — so roughly 9 builds/month before you'd
hit the free quota (after which GitHub blocks further LFS pulls until the quota
resets or you buy a data pack, a few dollars for more bandwidth). For normal use
(initial deploy + occasional redeploys after code changes) this is comfortably
enough; if you're iterating heavily, avoid triggering a rebuild on every trivial
push (e.g. batch commits, or use `npx vercel --prod` locally instead of git-triggered
deploys while testing).

## Local development

```bash
npm install
npm run dev
```

Open http://localhost:3000, paste a privacy-policy sentence, and click Classify. The
first request after starting the server is slow (the model loads into memory);
subsequent requests are fast (~15ms observed locally).

## Pushing to GitHub and deploying to Vercel

```bash
cd deploy
git lfs install                                   # once, if not already set up
git remote add origin <your-new-empty-repo-url>   # e.g. git@github.com:you/privacy-policy-classifier.git
git push -u origin master
```

Then in Vercel:

1. **Add New Project** → import the GitHub repo you just pushed to.
2. Framework preset: Next.js (auto-detected). Root Directory: leave as `.` (repo
  root), since this repo *is* the app — no need to set a subdirectory.
3. No environment variables are required.
4. **Before or right after the first deploy, enable Git LFS**: project **Settings** →
  **Git** → toggle **Git LFS** on. This is off by default on every new Vercel
  project and is *not* optional here — without it, Vercel checks out the ~130-byte
  LFS *pointer* file instead of the real 110MB ONNX weights, and every prediction
  fails at runtime with `Load model from .../model_quantized.onnx failed: Protobuf
  parsing failed` (confirmed the hard way on the first deploy of this app). Vercel
  requires a fresh redeploy after flipping this toggle — **Deployments** → **⋯** on
  the latest one → **Redeploy**.
5. The model files are bundled into the serverless function automatically via
  `outputFileTracingIncludes` in `next.config.js` — no manual upload step, and this
  has been verified locally (`npm run build` + inspecting the `.next` file trace
  confirms `models/**` and the correct `onnxruntime-node` native binary are included).

Or from the CLI, run from inside `deploy/`:

```bash
npx vercel        # preview deployment
npx vercel --prod # production deployment
```

## Regenerating the model (optional)

Only needed if you retrain or want to reproduce the checkpoint yourself. From the
project root, with the project's `.venv` activated:

```bash
python code/train_deploy_model.py                 # trains + saves deploy/model_raw/
pip install optimum-onnx onnx onnxruntime          # one-off, export-only deps
python code/export_deploy_model.py                 # writes deploy/models/bert-privacy-policy/
```

`train_deploy_model.py` retrains only the single winning configuration already
identified in `PROJECT_CONTEXT.md` §8.1 — it does not repeat the full §3.8
hyperparameter search, since that comparison is already recorded in
`logs/training_runs.csv`. `export_deploy_model.py` converts the PyTorch checkpoint to
ONNX and int8-quantizes it (~110 MB, down from ~440 MB fp32) so the model fits
comfortably inside Vercel's serverless function size limit.

## How it works

- `lib/classify.ts` — loads the local ONNX model once per warm serverless instance
  (module-level singleton) and runs the `text-classification` pipeline.
- `app/api/predict/route.ts` — POST `{ "text": "..." }` → `{ topLabel, topScore, scores }`.
  Runs on the Node.js runtime (not Edge), since inference needs filesystem access to
  the bundled weights and the `onnxruntime-node` native addon.
- `app/page.tsx` — the UI: a textarea, example clauses, and a bar chart of all 8
  category probabilities.
- `next.config.js` — two non-obvious settings that are load-bearing for a working
  production deploy: `outputFileTracingIncludes` (without it, Next.js's file tracer
  never discovers `models/**` since it's read via a dynamic `fs` path, not a static
  import, and Vercel's build silently omits it) and `serverExternalPackages`
  (without it, webpack tries to bundle `onnxruntime-node`'s native `.node` addon and
  breaks its runtime platform-detection).

## Limits and honesty notes

- This model was fine-tuned on English privacy-policy spans from the OPP-115 corpus
  (US company policies). Predictions on very different text (other languages, other
  domains) are not validated.
- Per `PROJECT_CONTEXT.md` decision 10: this tool is an exploratory research demo, not
  a compliance, legal, or safety verdict about any policy or company — the UI footer
  states this explicitly.
- The model only sees the single sentence/clause you paste, not surrounding context
  from the rest of a policy — matching how the classifier was trained (span-level
  input, per `PROJECT_CONTEXT.md` decision 1).
