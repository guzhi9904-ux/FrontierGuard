"""Free-generation evaluation for BF16, uniform quant and precision maps."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from frontierguard.io import read_json, read_jsonl, write_jsonl
from frontierguard.models.hf_runner import HFRunner, SamplingConfig
from frontierguard.quant.controller import instrument_linear_layers
from frontierguard.schemas import GenerationRecord, PrecisionMap
from frontierguard.traces.verify import extract_final_answer, verify_math_answer
from frontierguard.utils.reproducibility import stable_run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="JSONL: id, problem, answer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--precision-map")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()

    runner = HFRunner.from_pretrained(args.model)
    precision_map = None
    if args.precision_map:
        precision_map = PrecisionMap.from_dict(read_json(args.precision_map))
        runner.controller = instrument_linear_layers(
            runner.model, precision_map, materialize_weights=True
        )
        runner.kv_bits = precision_map.default.kv_bits
        runner.kv_group_size = precision_map.default.kv_group_size
        runner.kv_symmetric = precision_map.default.symmetric_kv
    run_id = stable_run_id(
        {
            "model": args.model,
            "condition": args.condition,
            "precision_map": precision_map.to_dict() if precision_map else None,
            "seeds": args.seeds,
            "max_new_tokens": args.max_new_tokens,
        }
    )

    records = []
    for row in read_jsonl(args.input):
        prompt_ids = runner.encode_chat(row["problem"])
        for seed in args.seeds:
            sampling = SamplingConfig(max_new_tokens=args.max_new_tokens, seed=seed)
            generated = runner.generate(prompt_ids, sampling)
            extracted = extract_final_answer(generated["text"])
            record = GenerationRecord(
                run_id=run_id,
                problem_id=str(row["id"]),
                condition=args.condition,
                seed=seed,
                output=generated["text"],
                extracted_answer=extracted,
                correct=verify_math_answer(extracted, str(row["answer"])),
                prompt_tokens=int(prompt_ids.shape[-1]),
                output_tokens=generated["output_tokens"],
                truncated=generated["truncated"],
                latency_seconds=generated["latency_seconds"],
            )
            records.append(asdict(record))
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
