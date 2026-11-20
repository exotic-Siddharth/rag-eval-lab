// Run the in-browser pipeline (rag.js + models.js) on the questions in reference.json and compare it with
// the Python pipeline's results. Writes parity.json (shown on the page) and fails if the port drifts.
//   node web/parity.mjs web/build
import { readFile, writeFile } from "node:fs/promises";
import { loadModels } from "./models.js";
import { createPipeline } from "./rag.js";

const dir = process.argv[2] ?? "web/build";
const read = async (f) => JSON.parse(await readFile(`${dir}/${f}`, "utf8"));
const [chunks, meta, ref] = await Promise.all([read("chunks.json"), read("meta.json"), read("reference.json")]);
const buf = await readFile(`${dir}/embeddings.bin`);
const embeddings = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const models = await loadModels();
const pipe = createPipeline({
  chunks, embeddings, dim: meta.browser.dim, embedQuery: models.embedQuery, scorePairs: models.scorePairs,
  retrieveK: meta.retrieve_k, pool: meta.pool, rerankTopN: meta.rerank_top_n, contextK: meta.context_k,
  groundingThreshold: meta.grounding_threshold,
});

const ids = (hits, k) => hits.slice(0, k).map(([i]) => i);
const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
const c = { bm25_top10_same: 0, dense_top1_same: 0, fused_overlap: 0, top1_same: 0, answer_same: 0,
            r1_py: 0, r1_js: 0, r3_py: 0, r3_js: 0 };
let nAns = 0, ms = 0;
const diffs = [];
for (const q of ref) {
  const t = Date.now();
  const r = await pipe.retrieve(q.question);
  const a = pipe.answer(q.question, r.reranked);
  ms += Date.now() - t;
  // compare scores, not ids: BM25 ties are ordered arbitrarily by numpy's argpartition
  c.bm25_top10_same += r.sparse.slice(0, 10).every(([, s], j) => q.bm25[j] && Math.abs(s - q.bm25[j][1]) < 2e-6);
  c.dense_top1_same += r.dense[0][0] === q.dense[0][0];
  const py = new Set(ids(q.hybrid, 20));
  c.fused_overlap += ids(r.candidates, 20).filter((i) => py.has(i)).length / Math.max(py.size, 1);
  c.top1_same += r.reranked[0][0] === q.reranked[0][0];
  c.answer_same += a.answer === q.answer;
  if (q.answerable && q.relevant.length) {
    nAns++;
    const rel = new Set(q.relevant);
    c.r1_py += rel.has(q.reranked[0][0]);
    c.r1_js += rel.has(r.reranked[0][0]);
    c.r3_py += ids(q.reranked, 3).some((i) => rel.has(i));
    c.r3_js += ids(r.reranked, 3).some((i) => rel.has(i));
  }
  if (r.reranked[0][0] !== q.reranked[0][0] && diffs.length < 8) {
    diffs.push(`  ${q.question}\n    python top: ${chunks[q.reranked[0][0]].title} #${q.reranked[0][0]} | browser top: ${chunks[r.reranked[0][0]].title} #${r.reranked[0][0]}`);
  }
}
const n = ref.length, f = (x, d) => Math.round((x / d) * 1000) / 1000;
const report = {
  questions: n, answerable: nAns,
  same_bm25_top10: f(c.bm25_top10_same, n), same_dense_top1: f(c.dense_top1_same, n),
  fused_top20_overlap: f(c.fused_overlap, n), same_top_passage: f(c.top1_same, n), same_answer: f(c.answer_same, n),
  recall_at_1: { python: f(c.r1_py, nAns), browser: f(c.r1_js, nAns) },
  recall_at_3: { python: f(c.r3_py, nAns), browser: f(c.r3_js, nAns) },
  node_ms_per_question: Math.round(ms / n),
  models: meta.browser, python: meta.python, built: new Date().toISOString().slice(0, 10),
};
await writeFile(`${dir}/parity.json`, JSON.stringify(report, null, 1));
console.log(JSON.stringify(report, null, 1));
if (diffs.length) console.log("Different top passage (first few):\n" + diffs.join("\n"));

const problems = [];
if (report.same_bm25_top10 < 0.95) problems.push("BM25 port differs from Python");
if (report.same_top_passage < 0.85) problems.push("top passage differs from Python too often");
if (report.recall_at_1.browser < report.recall_at_1.python - 0.05) problems.push("Recall@1 dropped by more than 0.05");
if (problems.length) { console.error("PARITY CHECK FAILED: " + problems.join("; ")); process.exit(1); }
console.log("parity check passed");
