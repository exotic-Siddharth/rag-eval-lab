---
title: rag-eval-lab
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RAG over Wikipedia with reranking and guardrails
---

# rag-eval-lab: live demo

Ask questions about 35 Wikipedia articles (the SQuAD 2.0 dev set). Each question goes through
hybrid BM25 + bge-small retrieval, a MiniLM cross-encoder reranker, prompt-injection and grounding
guardrails, and a Qwen3-0.6B reader (or an extractive no-LLM baseline).

The page shows the retrieved passages, the answer or the reason the system abstained, SQuAD's expected
answer for the example questions, and the time spent in each stage.

**Source code and full evaluation:** [github.com/exotic-Siddharth/rag-eval-lab](https://github.com/exotic-Siddharth/rag-eval-lab)

Built by [Siddharth Patel](https://exotic-siddharth.github.io). Questions and articles from
[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0).
