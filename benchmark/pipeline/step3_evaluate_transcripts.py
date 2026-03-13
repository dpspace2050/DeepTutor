#!/usr/bin/env python3
"""
Step 3: Evaluate transcripts for specified KBs.

Only evaluates transcripts that belong to input KB names.
Missing transcripts are recorded as errors (continue-on-error).
Supports resume by default: existing evaluation outputs are skipped unless --force.

Input:
  Expected profile set from:
    <output_root>/entries/<kb_name>/profiles/<profile_id>/entries.jsonl
  Transcript:
    <output_root>/transcripts/<kb_name>/<backend>/<profile_id>.json

Output:
  <output_root>/evaluations/<kb_name>/<backend>/<profile_id>_eval.json

Manifest:
  <output_root>/manifests/step3_manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluation.evaluator import evaluate_transcript

logger = logging.getLogger("benchmark.pipeline.step3")

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "data" / "bench_pipeline"


def _parse_names(raw: str) -> list[str]:
    return sorted(set(n.strip() for n in raw.split(",") if n.strip()))


async def _evaluate_one_transcript(
    *,
    kb_name: str,
    profile_id: str,
    backend: str,
    transcript_path: Path,
    output_root: Path,
    temperature: float,
    skip_turns: bool,
    force: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        out_path = output_root / "evaluations" / kb_name / backend / f"{profile_id}_eval.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "kb_name": kb_name,
            "profile_id": profile_id,
            "backend": backend,
            "transcript_path": str(transcript_path),
            "evaluation_path": str(out_path),
            "status": "ok",
            "error": None,
        }
        if out_path.exists() and not force:
            record["skipped_existing"] = True
            logger.info("[%s/%s] %s eval exists, skipped", kb_name, backend, profile_id)
            return record
        try:
            result = await evaluate_transcript(
                transcript_path=transcript_path,
                skip_turns=skip_turns,
                temperature=temperature,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)
            logger.error("[%s/%s] %s eval failed: %s", kb_name, backend, profile_id, e)
        return record


def _extract_eval_summary(eval_data: dict) -> dict:
    agg = eval_data.get("aggregate", {})
    summary = {
        "turn_count": agg.get("turn_count", {}),
        "source_faithfulness": agg.get("source_faithfulness", {}),
        "teaching_quality": agg.get("teaching_quality", {}),
    }
    pq = agg.get("practice_questions")
    if pq:
        summary["practice_questions"] = pq
    return summary


def _safe_avg(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def _build_aggregate_summary(results: list[dict], output_root: Path) -> dict:
    grouped: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        backend = r.get("backend", "unknown")
        grouped.setdefault(
            backend,
            {
                "num_profiles": 0,
                "paired_turns_total": 0,
                "tutor_turns_total": 0,
                "faithfulness_scores": [],
                "personalization_scores": [],
                "applicability_scores": [],
                "vividness_scores": [],
                "logical_depth_scores": [],
                "pq_total_questions": 0,
                "pq_fitness": [],
                "pq_groundedness": [],
                "pq_diversity": [],
                "pq_answer_quality": [],
                "pq_cross_concept": [],
            },
        )
        try:
            with open(r["evaluation_path"], encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            continue
        s = _extract_eval_summary(eval_data)
        g = grouped[backend]
        g["num_profiles"] += 1
        g["paired_turns_total"] += (
            s.get("turn_count", {}).get("paired_turns_total", 0)
        )
        g["tutor_turns_total"] += (
            s.get("turn_count", {}).get("tutor_turns_total", 0)
        )
        faith = s.get("source_faithfulness", {}).get("avg_score_overall")
        if isinstance(faith, (int, float)):
            g["faithfulness_scores"].append(float(faith))
        personalization = s.get("teaching_quality", {}).get("avg_personalization_overall")
        if isinstance(personalization, (int, float)):
            g["personalization_scores"].append(float(personalization))
        app = s.get("teaching_quality", {}).get("avg_applicability_overall")
        if isinstance(app, (int, float)):
            g["applicability_scores"].append(float(app))
        vividness = s.get("teaching_quality", {}).get("avg_vividness_overall")
        if isinstance(vividness, (int, float)):
            g["vividness_scores"].append(float(vividness))
        logical_depth = s.get("teaching_quality", {}).get("avg_logical_depth_overall")
        if isinstance(logical_depth, (int, float)):
            g["logical_depth_scores"].append(float(logical_depth))

        pq = s.get("practice_questions", {})
        if pq:
            g["pq_total_questions"] += pq.get("total_questions_across_sessions", 0)
            for key, lst_key in [
                ("avg_fitness", "pq_fitness"),
                ("avg_groundedness", "pq_groundedness"),
                ("avg_diversity", "pq_diversity"),
                ("avg_answer_quality", "pq_answer_quality"),
                ("avg_cross_concept", "pq_cross_concept"),
            ]:
                v = pq.get(key)
                if isinstance(v, (int, float)):
                    g[lst_key].append(float(v))

    out: dict[str, dict] = {}
    for backend, s in grouped.items():
        paired_turns = s["paired_turns_total"]
        tutor_turns = s["tutor_turns_total"]
        backend_summary: dict[str, Any] = {
            "num_profiles": s["num_profiles"],
            "paired_turns_total": paired_turns,
            "tutor_turns_total": tutor_turns,
            "avg_faithfulness": _safe_avg(s["faithfulness_scores"]),
            "avg_personalization": _safe_avg(s["personalization_scores"]),
            "avg_applicability": _safe_avg(s["applicability_scores"]),
            "avg_vividness": _safe_avg(s["vividness_scores"]),
            "avg_logical_depth": _safe_avg(s["logical_depth_scores"]),
        }
        if s["pq_total_questions"] > 0:
            backend_summary["practice_questions"] = {
                "total_questions": s["pq_total_questions"],
                "avg_fitness": _safe_avg(s["pq_fitness"]),
                "avg_groundedness": _safe_avg(s["pq_groundedness"]),
                "avg_diversity": _safe_avg(s["pq_diversity"]),
                "avg_answer_quality": _safe_avg(s["pq_answer_quality"]),
                "avg_cross_concept": _safe_avg(s["pq_cross_concept"]),
            }
        out[backend] = backend_summary
    return {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(output_root),
        "by_backend": out,
    }


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Step3: evaluate transcripts")
    parser.add_argument("--kb-names", required=True, help="Comma-separated KB names")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Pipeline output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--backends",
        default="mock,deep_tutor",
        help=(
            "Comma-separated backends to evaluate "
            "(e.g., mock,deep_tutor,deep_tutor_no_rag,deep_tutor_no_memory,"
            "deep_tutor_no_rag_memory)"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Max parallel transcript evaluations",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature for evaluation",
    )
    parser.add_argument(
        "--skip-turns",
        action="store_true",
        help="Skip per-turn LLM metrics and keep only turn_count",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation and overwrite existing outputs.",
    )
    args = parser.parse_args()

    kb_names = _parse_names(args.kb_names)
    backends = _parse_names(args.backends)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    entries_root = output_root / "entries"
    transcripts_root = output_root / "transcripts"
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    print(f"KBs: {len(kb_names)} | Backends: {backends}")
    print(f"Concurrency(transcript): {args.concurrency}")
    print(
        f"Resume mode: {'disabled (--force)' if args.force else 'enabled (skip existing evaluations)'}"
    )
    print(f"Output root: {output_root}")

    missing_errors: list[dict] = []
    tasks = []
    sem = asyncio.Semaphore(args.concurrency)

    for kb_name in kb_names:
        profiles_root = entries_root / kb_name / "profiles"
        if not profiles_root.exists():
            missing_errors.append(
                {
                    "kb_name": kb_name,
                    "profile_id": None,
                    "backend": None,
                    "error": f"Missing entries root for KB: {profiles_root}",
                }
            )
            logger.error("[%s] missing entries root: %s", kb_name, profiles_root)
            continue

        profile_ids = sorted(p.name for p in profiles_root.iterdir() if p.is_dir())
        for profile_id in profile_ids:
            for backend in backends:
                transcript_path = transcripts_root / kb_name / backend / f"{profile_id}.json"
                if not transcript_path.exists():
                    missing_errors.append(
                        {
                            "kb_name": kb_name,
                            "profile_id": profile_id,
                            "backend": backend,
                            "error": f"Missing transcript: {transcript_path}",
                        }
                    )
                    logger.error(
                        "[%s/%s] %s missing transcript", kb_name, backend, profile_id
                    )
                    continue
                tasks.append(
                    _evaluate_one_transcript(
                        kb_name=kb_name,
                        profile_id=profile_id,
                        backend=backend,
                        transcript_path=transcript_path,
                        output_root=output_root,
                        temperature=args.temperature,
                        skip_turns=args.skip_turns,
                        force=args.force,
                        semaphore=sem,
                    )
                )

    logger.info("Launching %d transcript evaluation tasks", len(tasks))
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    eval_results = []
    task_errors = 0
    skipped_existing = 0
    for r in task_results:
        if isinstance(r, Exception):
            task_errors += 1
            eval_results.append({"status": "error", "error": str(r)})
        else:
            eval_results.append(r)
            if r.get("status") != "ok":
                task_errors += 1
            if r.get("skipped_existing"):
                skipped_existing += 1

    newly_evaluated = len(eval_results) - skipped_existing

    aggregate = _build_aggregate_summary(eval_results, output_root)
    summary_path = manifests_root / "step3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    manifest = {
        "step": "step3_evaluate_transcripts",
        "timestamp": datetime.now().isoformat(),
        "kb_names": kb_names,
        "backends": backends,
        "output_root": str(output_root),
        "concurrency_transcript": args.concurrency,
        "overwrite": bool(args.force),
        "missing_errors": missing_errors,
        "results": eval_results,
        "num_evaluated": len(eval_results),
        "num_skipped_existing": skipped_existing,
        "num_newly_evaluated": newly_evaluated,
        "num_errors": len(missing_errors) + task_errors,
        "summary_path": str(summary_path),
    }
    manifest_path = manifests_root / "step3_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSummary: {summary_path}")
    print(f"Manifest: {manifest_path}")
    print(
        f"Done. Evaluated: {manifest['num_evaluated']} "
        f"(new: {manifest['num_newly_evaluated']}, skipped: {manifest['num_skipped_existing']}) | "
        f"Errors: {manifest['num_errors']}"
    )
    if manifest["num_errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
