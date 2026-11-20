// ONNX versions of the two models raglab uses, run with transformers.js (browser: WASM; Node: onnxruntime-node).
// In the browser "@huggingface/transformers" resolves through the page's import map.
import { AutoModelForSequenceClassification, AutoTokenizer, pipeline } from "@huggingface/transformers";

export const EMBED_MODEL = "Xenova/bge-small-en-v1.5"; // BAAI/bge-small-en-v1.5
export const RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"; // cross-encoder/ms-marco-MiniLM-L-6-v2
export const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
export const DTYPE = "q8"; // 8-bit weights: about 57 MB for both models instead of about 220 MB

export async function loadModels({ dtype = DTYPE, onProgress } = {}) {
  const opts = { dtype, progress_callback: onProgress };
  const [embedder, tokenizer, crossEncoder] = await Promise.all([
    pipeline("feature-extraction", EMBED_MODEL, opts),
    AutoTokenizer.from_pretrained(RERANK_MODEL, { progress_callback: onProgress }),
    AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, opts),
  ]);

  /** Unit-length CLS embeddings (as sentence-transformers does for bge), one row per text. */
  async function embed(texts) {
    const out = await embedder(texts, { pooling: "cls", normalize: true });
    return out.data; // Float32Array of texts.length * dim
  }

  /** Cross-encoder relevance logits for (query, passage) pairs. */
  async function scorePairs(query, passages, batchSize = 10) {
    const scores = [];
    for (let i = 0; i < passages.length; i += batchSize) {
      const batch = passages.slice(i, i + batchSize);
      const inputs = tokenizer(new Array(batch.length).fill(query), { text_pair: batch, padding: true, truncation: true });
      const { logits } = await crossEncoder(inputs);
      scores.push(...Array.from(logits.data));
    }
    return scores;
  }

  return {
    embed,
    embedQuery: async (q) => embed([QUERY_PREFIX + q]),
    scorePairs,
  };
}
