"use client";

import { useState } from "react";

const CATEGORY_MEANINGS: Record<string, string> = {
  "First Party Collection/Use": "Direct collection or use of data by the website itself",
  "Third Party Sharing/Collection": "Data shared with, or collected by, third parties",
  "User Choice/Control": "Opt-out, preferences, or consent mechanisms",
  "Data Security": "Protection, encryption, or security measures",
  "User Access, Edit and Deletion": "Rights to access, modify, or delete personal data",
  "International and Specific Audiences": "GDPR, CCPA, children's data, or regional handling",
  "Policy Change": "Notification and procedure for policy updates",
  "Data Retention": "Storage duration and deletion timelines",
};

const EXAMPLES = [
  "We may share your personal information with advertising partners for marketing purposes.",
  "You may opt out of receiving promotional emails at any time by clicking the unsubscribe link.",
  "We use industry-standard encryption to protect your data from unauthorized access.",
  "We retain your account information for up to 90 days after you close your account.",
  "This policy applies to users in the European Union under the GDPR.",
];

interface CategoryScore {
  label: string;
  score: number;
}

interface ClassificationResult {
  topLabel: string;
  topScore: number;
  scores: CategoryScore[];
}

export default function Home() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Prediction failed.");
      setResult(data as ClassificationResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Privacy Policy Clause Classifier</h1>
      <p className="subtitle">
        Paste a sentence from a privacy policy and a fine-tuned BERT model predicts which of
        8 data-practice categories it describes. Trained on the OPP-115 corpus (CSE440
        project — Privacy Practice Category Classification).
      </p>

      <form onSubmit={handleSubmit}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. We may share your personal information with advertising partners for marketing purposes."
          maxLength={2000}
        />

        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button
              type="button"
              key={ex}
              className="example-chip"
              onClick={() => setText(ex)}
            >
              {ex.length > 42 ? ex.slice(0, 42) + "…" : ex}
            </button>
          ))}
        </div>

        <div className="submit-row">
          <button type="submit" className="primary" disabled={!text.trim() || loading}>
            {loading ? "Classifying…" : "Classify"}
          </button>
          {error && <span className="error">{error}</span>}
        </div>
      </form>

      {result && (
        <div className="result-card">
          <div className="result-top-label">Predicted category</div>
          <div className="result-top-value">{result.topLabel}</div>
          <div className="result-top-meaning">
            {CATEGORY_MEANINGS[result.topLabel] ?? ""} — {(result.topScore * 100).toFixed(1)}%
            confidence
          </div>

          {result.scores.map((s) => (
            <div key={s.label}>
              <div className="score-row">
                <span>{s.label}</span>
                <span className="score-pct">{(s.score * 100).toFixed(1)}%</span>
              </div>
              <div className="score-bar-track">
                <div
                  className="score-bar-fill"
                  style={{ width: `${Math.max(s.score * 100, 1.5)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      <footer>
        Model: BERT Base (bert-base-uncased, partially fine-tuned — top 2 of 12 encoder
        layers + classifier head), evaluated under a policy-disjoint train/test split to
        avoid boilerplate leakage. Test macro-F1 0.6987, accuracy 0.7159 on OPP-115. This
        tool is an exploratory research demo, not a compliance, legal, or safety verdict
        about any policy or company.
      </footer>
    </main>
  );
}
