"""Re-judge saved counterfactual continuations without loading a model."""

from __future__ import annotations

import argparse
from frontierguard import __version__
from frontierguard.frontier.pipeline import TeacherForcedScan
from frontierguard.frontier.rejudge import rejudge_counterfactual
from frontierguard.io import read_jsonl, write_jsonl
from frontierguard.traces.segment import segment_reasoning
from frontierguard.workflows import complete_frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="saved 03b frontier JSONL")
    parser.add_argument("--traces", required=True, help="source BF16 trace JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-trustworthy-seeds", type=int, default=4)
    args = parser.parse_args()
    if args.input == args.output:
        raise ValueError("--output must differ from --input")

    traces = {str(row["problem_id"]): row for row in read_jsonl(args.traces)}
    rows = []
    for saved in read_jsonl(args.input):
        problem_id = str(saved["problem_id"])
        if problem_id not in traces:
            raise KeyError(f"{problem_id} is missing from {args.traces}")
        trace = traces[problem_id]
        # Reproduce v0.3 phase-aware character boundaries even when the source
        # trace stores legacy, presentation-inclusive steps.
        steps = segment_reasoning(trace["response"])
        counterfactual = rejudge_counterfactual(
            saved["counterfactual"],
            response=trace["response"],
            steps=steps,
            reference_answer=str(trace["reference_answer"]),
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            min_trustworthy_seeds=args.min_trustworthy_seeds,
        )
        scan = TeacherForcedScan(
            token_signals=None,
            step_indices=[int(item) for item in saved["step_indices"]],
            step_jsd=[float(item) for item in saved["step_jsd"]],
            step_margin_drop=[float(item) for item in saved["step_margin_drop"]],
            step_nll_gap=[float(item) for item in saved["step_nll_gap"]],
            shortlist=[int(item) for item in saved.get("shortlist", [])],
        )
        refreshed = {
            **saved,
            **complete_frontier(scan, counterfactual),
            "frontierguard_version": __version__,
            "reference_answer": str(trace["reference_answer"]),
            "counterfactual": counterfactual,
            "rejudged": True,
            "rejudged_from_version": saved.get("frontierguard_version"),
        }
        quant_summary = counterfactual["condition_summaries"]["quantized"]
        if quant_summary["successes"] == 0:
            refreshed["counterfactual_status"] = "all_quantized_fail"
        elif quant_summary["successes"] == quant_summary["rollouts"]:
            refreshed["counterfactual_status"] = "all_quantized_succeed"
        else:
            refreshed["counterfactual_status"] = "partially_recoverable"
        rows.append(refreshed)
        write_jsonl(args.output, rows)
        print(
            f"[{len(rows)}] {problem_id}: "
            f"first_error={refreshed['first_error_step']} "
            f"recovery={refreshed['recovery_frontier_step']} "
            f"window={refreshed['frontier_window']}",
            flush=True,
        )
    print(f"wrote {len(rows)} re-judged result(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
