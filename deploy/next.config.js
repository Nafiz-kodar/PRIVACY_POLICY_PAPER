const path = require("node:path");

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Without this, Next.js auto-detects the workspace root by walking up for a
  // lockfile and can land on an unrelated one outside this project (it did in
  // dev: C:\Users\<user>\package-lock.json), which then silently changes what
  // "./models/**/*" below resolves relative to.
  outputFileTracingRoot: path.join(__dirname),
  // The fine-tuned BERT weights (deploy/models/**) are read from disk at request
  // time via env.localModelPath, not imported statically, so Next.js's default
  // file-tracing never discovers them. Without this, Vercel's build omits the
  // model files from the serverless function bundle and every prediction 404s
  // at runtime in production even though `next dev` works locally.
  outputFileTracingIncludes: {
    "/api/predict": ["./models/**/*"],
  },
  // onnxruntime-node ships a native .node addon selected at runtime based on
  // process.platform/process.arch; letting webpack bundle it breaks that
  // resolution. Next.js 15's stable equivalent of the old
  // experimental.serverComponentsExternalPackages — leave both packages as
  // plain `require()`s resolved from node_modules instead.
  serverExternalPackages: ["onnxruntime-node", "sharp"],
};

module.exports = nextConfig;
