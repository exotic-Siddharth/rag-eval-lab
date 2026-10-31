**extractive** | paragraph | hybrid(bm25+dense:bge-small-en-v1.5) | reranker: rerank:ms-marco-MiniLM-L-6-v2@20 | top-3 chunks in context

Test: 400 questions (half unanswerable). Thresholds tuned on 150 tune-split questions.

| setting | EM | F1 | HasAns F1 | NoAns acc | answer rate | generations skipped |
|---|---|---|---|---|---|---|
| reference: abstain on everything | 0.500 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 |
| no guardrails | 0.003 | 0.090 | 0.180 | 0.000 | 1.000 | 0.000 |
| confidence gate (pass 50%) | 0.405 | 0.444 | 0.083 | 0.805 | 0.290 | 0.710 |
| grounding check (>= 0.0) | 0.003 | 0.090 | 0.180 | 0.000 | 1.000 | 0.000 |
| gate (pass 50%) + grounding (>= 0.0) | 0.405 | 0.444 | 0.083 | 0.805 | 0.290 | 0.710 |

- Context recall (gold span inside the 3 chunks given to the generator): 0.960
- HasAns F1 when the gold chunk was retrieved: 0.184; when it was missed: 0.096
- Confidence-gate AUROC (answerable vs unanswerable): 0.629
- Latency: retrieval p50 1317 ms / p95 2702 ms; generation p50 0 ms / p95 0 ms
- Tokens per query: 0 in / 0.0 out; API cost per 1k queries: $0.00
