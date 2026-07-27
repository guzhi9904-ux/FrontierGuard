"""Generate verified BF16 traces from a problem JSONL file."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.traces.build import build_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="JSONL: id, problem, answer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    runner = HFRunner.from_pretrained(args.model)
    traces = []
    for row in read_jsonl(args.input):
        prompt_ids = runner.encode_chat(row["problem"])
        for seed in args.seeds:
            sampling = SamplingConfig(max_new_tokens=args.max_new_tokens, seed=seed)
            generated = runner.generate(prompt_ids, sampling)
            trace = build_trace(
                problem_id=str(row["id"]),
                problem=row["problem"],
                response=generated["text"],
                reference_answer=str(row["answer"]),
                model_id=args.model,
                model_revision=None,
                seed=seed,
                generation_config=asdict(sampling),
                tokenizer=runner.tokenizer,
                token_ids=generated["token_ids"],
                truncated=generated["truncated"],
            )
            if trace.correct:
                traces.append(trace)
    write_jsonl(args.output, traces)


if __name__ == "__main__":
    main()
