"""Generate verified BF16 traces from a problem JSONL file."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from frontierguard.io import read_jsonl, write_json, write_jsonl
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.quant.calibration import file_sha256
from frontierguard.traces.build import build_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--input", required=True, help="JSONL: id, problem, answer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-output", help="optional JSONL including incorrect traces")
    parser.add_argument("--summary-output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("input contains no problems")
    dataset_hash = file_sha256(args.input)
    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    traces = []
    all_traces = []
    truncated = 0
    generations = 0
    for row_index, row in enumerate(rows, start=1):
        prompt_ids = runner.encode_chat(row["problem"])
        for seed in args.seeds:
            generations += 1
            sampling = SamplingConfig(max_new_tokens=args.max_new_tokens, seed=seed)
            generated = runner.generate(prompt_ids, sampling)
            trace = build_trace(
                problem_id=str(row["id"]),
                problem=row["problem"],
                response=generated["text"],
                reference_answer=str(row["answer"]),
                model_id=args.model,
                model_revision=args.revision,
                seed=seed,
                generation_config=asdict(sampling),
                tokenizer=runner.tokenizer,
                token_ids=generated["token_ids"],
                truncated=generated["truncated"],
                dataset_hash=dataset_hash,
            )
            truncated += int(trace.truncated)
            if args.all_output:
                all_traces.append(trace)
            if trace.correct:
                traces.append(trace)
            print(
                f"[{row_index}/{len(rows)}] id={row['id']} seed={seed} "
                f"correct={trace.correct} tokens={len(generated['token_ids'])} "
                f"truncated={trace.truncated}",
                flush=True,
            )
    write_jsonl(args.output, traces)
    if args.all_output:
        write_jsonl(args.all_output, all_traces)
    correct_problem_ids = {trace.problem_id for trace in traces}
    summary = {
        "model": args.model,
        "model_revision": args.revision,
        "input": str(Path(args.input).resolve()),
        "input_sha256": dataset_hash,
        "problems": len(rows),
        "seeds": args.seeds,
        "generations": generations,
        "correct_generations": len(traces),
        "correct_generation_rate": len(traces) / generations,
        "correct_problems": len(correct_problem_ids),
        "correct_problem_rate": len(correct_problem_ids) / len(rows),
        "truncated_generations": truncated,
        "trace_output": str(Path(args.output).resolve()),
        "all_output": str(Path(args.all_output).resolve()) if args.all_output else None,
    }
    summary_output = args.summary_output or f"{args.output}.summary.json"
    write_json(summary_output, summary)
    print(f"wrote {len(traces)} verified traces to {args.output}")
    print(f"wrote capability summary to {summary_output}")


if __name__ == "__main__":
    main()
