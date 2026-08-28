import { NextRequest, NextResponse } from "next/server";
import { classifyClause } from "@/lib/classify";

// Node.js runtime (not Edge): inference needs fs access to the bundled ONNX
// weights and onnxruntime-node's native binding, neither of which the Edge
// runtime supports.
export const runtime = "nodejs";
export const maxDuration = 30;

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be JSON." }, { status: 400 });
  }

  const text = (body as { text?: unknown })?.text;
  if (typeof text !== "string" || !text.trim()) {
    return NextResponse.json({ error: "`text` must be a non-empty string." }, { status: 400 });
  }

  try {
    const result = await classifyClause(text);
    return NextResponse.json(result);
  } catch (err) {
    console.error("Classification failed:", err);
    return NextResponse.json({ error: "Classification failed on the server." }, { status: 500 });
  }
}
