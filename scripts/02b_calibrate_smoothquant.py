"""Calibrate per-linear SmoothQuant-style input scales."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from frontierguard.io import read_jsonl
from frontierguard.models.hf_runner import HFRunner
from frontierguard.quant.calibration import (
    ActivationMaxCollector,
    CalibrationArtifact,
    build_smoothquant_scales,
    calibration_mode,
    file_sha256,
)


def _calibration_ids(runner: HFRunner, row: dict, max_seq_tokens: int) -> torch.Tensor:
    prompt_ids = runner.encode_chat(row["problem"])
    response = row.get("response")
    if response:
        response_ids = runner.tokenizer(
            response,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(runner.device)
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
    else:
        input_ids = prompt_ids
    return input_ids[:, :max_seq_tokens]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="disjoint QCal JSONL; accepts raw problems or generated trace rows",
    )
    parser.add_argument("--output", required=True, help="output .safetensors artifact")
    parser.add_argument("--revision")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-seq-tokens", type=int, default=2048)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))[: args.max_samples]
    if not rows:
        raise RuntimeError("calibration input contains no rows")

    runner = HFRunner.from_pretrained(args.model, revision=args.revision)
    with ActivationMaxCollector(runner.model) as collector:
        with calibration_mode(runner.model):
            for index, row in enumerate(rows, start=1):
                input_ids = _calibration_ids(runner, row, args.max_seq_tokens)
                runner.model(input_ids=input_ids, use_cache=False)
                print(
                    f"[{index}/{len(rows)}] calibrated {row.get('id', row.get('problem_id'))} "
                    f"tokens={input_ids.shape[-1]}",
                    flush=True,
                )

    maxima = collector.cpu_maxima()
    scales = build_smoothquant_scales(
        runner.model,
        maxima,
        alpha=args.alpha,
    )
    artifact = CalibrationArtifact(
        scales=scales,
        metadata={
            "method": "smoothquant_per_linear",
            "alpha": str(args.alpha),
            "model": args.model,
            "model_revision": args.revision or "unlocked",
            "source": str(Path(args.input).resolve()),
            "source_sha256": file_sha256(args.input),
            "samples": str(len(rows)),
            "max_seq_tokens": str(args.max_seq_tokens),
            "modules": str(len(scales)),
        },
    )
    artifact.save(args.output)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "samples": len(rows),
                "modules": len(scales),
                "alpha": args.alpha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
