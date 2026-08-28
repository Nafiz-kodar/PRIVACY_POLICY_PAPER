import path from "node:path";
import { env, pipeline, type TextClassificationPipeline } from "@huggingface/transformers";

// Model files live at deploy/models/bert-privacy-policy/ (config.json, tokenizer.*,
// onnx/model_quantized.onnx) — a locally fine-tuned checkpoint, never fetched from
// the Hugging Face Hub, so inference works offline and deterministically.
env.allowRemoteModels = false;
env.localModelPath = path.join(process.cwd(), "models") + path.sep;

const MODEL_ID = "bert-privacy-policy";

let pipelinePromise: Promise<TextClassificationPipeline> | null = null;

function getPipeline(): Promise<TextClassificationPipeline> {
  if (!pipelinePromise) {
    pipelinePromise = pipeline("text-classification", MODEL_ID, {
      dtype: "q8",
    }) as Promise<TextClassificationPipeline>;
  }
  return pipelinePromise;
}

export interface CategoryScore {
  label: string;
  score: number;
}

export interface ClassificationResult {
  topLabel: string;
  topScore: number;
  scores: CategoryScore[];
}

const MAX_INPUT_CHARS = 2000;

export async function classifyClause(text: string): Promise<ClassificationResult> {
  const trimmed = text.trim().slice(0, MAX_INPUT_CHARS);
  if (!trimmed) {
    throw new Error("Text is empty.");
  }

  const classifier = await getPipeline();
  const raw = await classifier(trimmed, { top_k: null });

  // top_k: null returns every class score for a single input as a flat array.
  const scores = (raw as unknown as CategoryScore[])
    .map((r) => ({ label: r.label, score: r.score }))
    .sort((a, b) => b.score - a.score);

  return {
    topLabel: scores[0].label,
    topScore: scores[0].score,
    scores,
  };
}
