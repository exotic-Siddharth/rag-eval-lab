"""Command-line entry point: `python -m raglab <command>`."""

from __future__ import annotations

import argparse
import json

from .data import download_squad
from .pipeline import build_pipeline, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="raglab", description="RAG with a measurable evaluation harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download SQuAD 2.0 dev")
    p.add_argument("--path", default="data/dev-v2.0.json")

    p = sub.add_parser("eval-retrieval", help="chunker x retriever x reranker grid")
    p.add_argument("--config", default="configs/retrieval_grid.yaml")
    p.add_argument("--out", default="results")

    p = sub.add_parser("eval-e2e", help="end-to-end eval with guardrail ablation")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--generator", choices=["extractive", "local", "claude"], help="override the config's generator")
    p.add_argument("--tag", help="name for the results files")
    p.add_argument("--out", default="results")

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question")
    p.add_argument("--config", default="configs/default.yaml")

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.cmd == "download":
        print(download_squad(args.path))
    elif args.cmd == "eval-retrieval":
        from .evaluate import run_retrieval_grid

        run_retrieval_grid(load_config(args.config), args.out)
    elif args.cmd == "eval-e2e":
        from .evaluate import run_e2e

        cfg = load_config(args.config)
        override = None
        if args.generator:
            override = cfg.get("generators", {}).get(args.generator, {"type": args.generator})
        res = run_e2e(cfg, args.out, override, args.tag)
        print(json.dumps(res["ablation_test"], indent=2))
    elif args.cmd == "ask":
        pipe, *_ = build_pipeline(load_config(args.config))
        print(json.dumps(pipe.ask(args.question).to_dict(), indent=2))
    elif args.cmd == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
