**local:Qwen3-0.6B** | paragraph | hybrid(bm25+dense:bge-small-en-v1.5) | reranker: rerank:ms-marco-MiniLM-L-6-v2@20 | top-3 chunks in context

Test: 400 questions (half unanswerable). Thresholds tuned on 150 tune-split questions.

| setting | EM | F1 | HasAns F1 | NoAns acc | answer rate | generations skipped |
|---|---|---|---|---|---|---|
| reference: abstain on everything | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 |
| no guardrails | 0.328 | 0.399 | 0.309 | 0.490 | 0.583 | 0.000 |
| confidence gate (pass 60%) | 0.460 | 0.496 | 0.168 | 0.825 | 0.260 | 0.630 |
| grounding check (>= 1.0) | 0.430 | 0.476 | 0.257 | 0.695 | 0.367 | 0.000 |
| gate (pass 60%) + grounding (>= 0.5) | 0.465 | 0.501 | 0.168 | 0.835 | 0.230 | 0.630 |

- Context recall (gold span inside the 3 chunks given to the generator): 0.960
- HasAns F1 when the gold chunk was retrieved: 0.315; when it was missed: 0.160
- Confidence-gate AUROC (answerable vs unanswerable): 0.629
- Latency: retrieval p50 2274 ms / p95 3905 ms; generation p50 5969 ms / p95 14945 ms
- Tokens per query: 676 in / 8.5 out; API cost per 1k queries: $0.00
