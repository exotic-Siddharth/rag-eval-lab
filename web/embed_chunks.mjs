// Embed every chunk with the same ONNX model and dtype the page uses for queries.
//   node web/embed_chunks.mjs web/build   ->  web/build/embeddings.bin (float32, row-major) + meta.json fields
import { readFile, writeFile } from "node:fs/promises";
import { DTYPE, EMBED_MODEL, QUERY_PREFIX, RERANK_MODEL, loadModels } from "./models.js";

const dir = process.argv[2] ?? "web/build";
const chunks = JSON.parse(await readFile(`${dir}/chunks.json`, "utf8"));
const models = await loadModels();
const texts = chunks.map((c) => `${c.title}. ${c.text}`); // as DenseRetriever.fit
const parts = [];
const t0 = Date.now();
for (let i = 0; i < texts.length; i += 32) parts.push(await models.embed(texts.slice(i, i + 32)));
const all = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
let off = 0;
for (const p of parts) { all.set(p, off); off += p.length; }
const dim = all.length / texts.length;
if (!Number.isInteger(dim)) throw new Error(`bad embedding size ${all.length} for ${texts.length} texts`);
await writeFile(`${dir}/embeddings.bin`, Buffer.from(all.buffer));

const meta = JSON.parse(await readFile(`${dir}/meta.json`, "utf8"));
meta.browser = { embed_model: EMBED_MODEL, rerank_model: RERANK_MODEL, dtype: DTYPE, query_prefix: QUERY_PREFIX, dim };
await writeFile(`${dir}/meta.json`, JSON.stringify(meta));
console.log(`embedded ${texts.length} chunks (dim ${dim}) in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
