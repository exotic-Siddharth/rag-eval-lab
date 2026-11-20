// The raglab retrieval pipeline ported to JavaScript, so the demo runs in a browser with no server.
// Every function mirrors its Python counterpart (named in the comments); web/parity.mjs checks the
// port against the Python pipeline on held-out questions.

// raglab/text.py
export const STOPWORDS = new Set(
  `a an and are as at be been but by can did do does for from had has have he her his how i if in
  into is it its many much not of on or she so than that the their them there these they this those
  to was were what when where which while who whom whose why will with would you`.split(/\s+/)
);
const WORD = /[\p{L}\p{N}_]+/gu; // Python's \w (letters, numbers, underscore)
const SENT_END = /(?<=[.!?])["')\]]*\s+(?=["'(\[]?[A-Z0-9])/g;

export function tokenize(text) {
  return (text.match(WORD) || []).map((t) => t.toLowerCase());
}

export function contentTokens(text) {
  return tokenize(text).filter((t) => !STOPWORDS.has(t));
}

/** Character spans [start, end) of sentences; blank lines always end a sentence. */
export function sentenceSpans(text) {
  const spans = [];
  for (const block of text.matchAll(/[^\n]+/g)) {
    const b0 = block.index, bt = block[0];
    let cur = 0;
    for (const m of bt.matchAll(SENT_END)) {
      spans.push([b0 + cur, b0 + m.index + m[0].trimEnd().length]);
      cur = m.index + m[0].length;
    }
    if (bt.slice(cur).trim()) spans.push([b0 + cur, b0 + bt.trimEnd().length]);
  }
  return spans.filter(([s, e]) => text.slice(s, e).trim());
}

// raglab/guardrails.py
const INJECTION = new RegExp([
  String.raw`ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)`,
  String.raw`disregard\s+(all\s+|the\s+)?(previous|prior|above|system)`,
  String.raw`\byou\s+are\s+now\s+(a|an|in)\b`,
  String.raw`reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)`,
  String.raw`<\s*/?\s*(system|instructions?)\s*>`,
  String.raw`\bnew\s+instructions\s*:`,
].join("|"), "i");

export const looksLikeInjection = (text) => INJECTION.test(text);

export function checkQuery(question, maxChars = 1000) {
  if (!question.trim()) return "empty_query";
  if (question.length > maxChars) return "query_too_long";
  if (looksLikeInjection(question)) return "prompt_injection";
  return null;
}

export function supportScore(answer, context) {
  const ans = contentTokens(answer);
  if (!ans.length) return 1.0;
  const ctx = new Set(contentTokens(context));
  return ans.filter((t) => ctx.has(t)).length / ans.length;
}

// raglab/retrievers.py
/** Indices of the k highest scores, highest first (ties: lower index first). */
export function topK(scores, k) {
  const idx = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a] || a - b);
  return idx.slice(0, k).map((i) => [i, scores[i]]);
}

/** Okapi BM25 over "title text", same weights as BM25Retriever. */
export class BM25 {
  constructor(chunks, k1 = 1.2, b = 0.75) {
    this.n = chunks.length;
    const docs = chunks.map((c) => {
      const counts = new Map();
      for (const t of tokenize(`${c.title} ${c.text}`)) counts.set(t, (counts.get(t) || 0) + 1);
      let len = 0;
      for (const v of counts.values()) len += v;
      return { counts, len };
    });
    const avg = docs.reduce((s, d) => s + d.len, 0) / this.n;
    const df = new Map();
    for (const d of docs) for (const t of d.counts.keys()) df.set(t, (df.get(t) || 0) + 1);
    this.postings = new Map();
    docs.forEach((d, i) => {
      for (const [t, tf] of d.counts) {
        const idf = Math.log1p((this.n - df.get(t) + 0.5) / (df.get(t) + 0.5));
        const w = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * d.len / avg));
        if (!this.postings.has(t)) this.postings.set(t, []);
        this.postings.get(t).push([i, w]);
      }
    });
  }

  search(query, k) {
    const scores = new Float64Array(this.n);
    for (const t of new Set(tokenize(query))) {
      for (const [i, w] of this.postings.get(t) || []) scores[i] += w;
    }
    return topK(scores, k);
  }
}

/** Cosine search over unit-length float32 embeddings stored row by row. */
export function denseSearch(queryVec, matrix, dim, k) {
  const n = matrix.length / dim, scores = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0, o = i * dim; j < dim; j++) s += queryVec[j] * matrix[o + j];
    scores[i] = s;
  }
  return topK(scores, k);
}

/** Reciprocal-rank fusion, same as HybridRetriever (insertion order breaks ties). */
export function rrf(lists, k, rrfK = 60) {
  const fused = new Map();
  for (const hits of lists) {
    hits.forEach(([idx], rank) => fused.set(idx, (fused.get(idx) || 0) + 1 / (rrfK + rank + 1)));
  }
  return [...fused.entries()].sort((a, b) => b[1] - a[1]).slice(0, k);
}

// raglab/generators.py: ExtractiveGenerator
export const NO_ANSWER = "NO_ANSWER";

export function extractiveAnswer(question, contexts) {
  const q = new Set(contentTokens(question));
  let best = "", bestScore = 0;
  contexts.forEach((c, rank) => {
    for (const [s, e] of sentenceSpans(c.text)) {
      const sent = c.text.slice(s, e);
      let overlap = 0;
      for (const t of new Set(contentTokens(sent))) if (q.has(t)) overlap++;
      const score = overlap - 0.01 * rank;
      if (overlap && score > bestScore) { best = sent; bestScore = score; }
    }
  });
  return best || NO_ANSWER;
}

/**
 * The full pipeline (RAGPipeline.ask with the extractive reader).
 * embedQuery(text) -> Float32Array; scorePairs(query, passages) -> number[] (cross-encoder scores).
 */
export function createPipeline({ chunks, embeddings, dim, embedQuery, scorePairs, retrieveK = 20, pool = 100,
                                 rerankTopN = 20, contextK = 3, groundingThreshold = 0.5 }) {
  const bm25 = new BM25(chunks);
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

  async function retrieve(question, latency = {}) {
    let t = now();
    const sparse = bm25.search(question, pool);
    const qv = await embedQuery(question);
    const dense = denseSearch(qv, embeddings, dim, pool);
    const candidates = rrf([sparse, dense], retrieveK);
    latency.retrieve_ms = now() - t;
    t = now();
    const top = candidates.slice(0, rerankTopN);
    const scores = await scorePairs(question, top.map(([i]) => `${chunks[i].title}. ${chunks[i].text}`));
    const reranked = top.map(([i], j) => [i, scores[j]]).sort((a, b) => b[1] - a[1]);
    latency.rerank_ms = now() - t;
    return { reranked, sparse, dense, candidates };
  }

  function answer(question, reranked, latency = {}) {
    const t = now();
    const kept = reranked.filter(([i]) => !looksLikeInjection(chunks[i].text));
    const contexts = kept.slice(0, contextK).map(([i]) => chunks[i]);
    const text = extractiveAnswer(question, contexts);
    latency.generate_ms = now() - t;
    const out = { answer: text, abstained: text === NO_ANSWER, reason: "ok", support: null,
                  kept, dropped: reranked.length - kept.length };
    if (out.abstained) return { ...out, reason: "model_abstained" };
    out.support = supportScore(text, contexts.map((c) => c.text).join(" "));
    if (groundingThreshold != null && out.support < groundingThreshold) {
      return { ...out, answer: NO_ANSWER, abstained: true, reason: "ungrounded" };
    }
    return out;
  }

  async function ask(question) {
    const latency = {};
    const blocked = checkQuery(question);
    if (blocked) return { blocked, latency };
    const r = await retrieve(question, latency);
    return { blocked: null, ...r, ...answer(question, r.reranked, latency), latency };
  }

  return { bm25, retrieve, answer, ask };
}
