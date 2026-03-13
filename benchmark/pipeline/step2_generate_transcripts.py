#!/usr/bin/env python3
"""
Step 2: Run simulations per profile, backend serially.

Input:
  <output_root>/entries/<kb_name>/profiles/<profile_id>/entries.jsonl

Output:
  <output_root>/transcripts/<kb_name>/<backend>/<profile_id>.json
  <output_root>/workspaces/<kb_name>/<backend>/<profile_id>/

Manifest:
  <output_root>/manifests/step2_manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.simulation.profile_evolver import evolve_profile as evolve_profile_fn

logger = logging.getLogger("benchmark.pipeline.step2")

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "data" / "bench_pipeline"
SUPPORTED_BACKENDS = {
    "mock",
    "deep_tutor",
    "deep_tutor_no_rag",
    "deep_tutor_no_memory",
    "deep_tutor_no_rag_memory",
}
_DIALOGUE_LOG_LOCK: asyncio.Lock | None = None


def _parse_names(raw: str) -> list[str]:
    return sorted(set(n.strip() for n in raw.split(",") if n.strip()))


def _load_entries(entries_jsonl: Path) -> list[dict]:
    entries: list[dict] = []
    with open(entries_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _load_existing_transcript(transcript_path: Path) -> dict | None:
    if not transcript_path.exists():
        return None
    try:
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.warning("Failed to read existing transcript %s: %s", transcript_path, e)
    return None


def _get_dialogue_log_lock() -> asyncio.Lock:
    global _DIALOGUE_LOG_LOCK
    if _DIALOGUE_LOG_LOCK is None:
        _DIALOGUE_LOG_LOCK = asyncio.Lock()
    return _DIALOGUE_LOG_LOCK


async def _emit_dialogue_log(
    *,
    kb_name: str,
    backend: str,
    profile_id: str,
    session_num: int,
    session_total: int,
    text: str,
) -> None:
    """Emit captured student/tutor dialogue lines with stable prefixes."""
    prefix = f"[{kb_name}/{backend}] {profile_id} session {session_num}/{session_total}"
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return
    async with _get_dialogue_log_lock():
        logger.info("%s dialogue begin", prefix)
        for line in lines:
            logger.info("%s %s", prefix, line)
        logger.info("%s dialogue end", prefix)


async def _simulate_profile_backend(
    *,
    kb_name: str,
    profile_id: str,
    entries: list[dict],
    backend: str,
    output_root: Path,
    max_turns: int,
    language: str,
    evolve_profile: bool,
    verbose: bool,
    force: bool,
) -> dict:
    from benchmark.simulation.conversation import (
        _run_single_session,
        _summarize_session,
    )

    if not entries:
        logger.warning("[%s/%s] %s has 0 entries, skipping", kb_name, backend, profile_id)
        return {
            "status": "skipped",
            "backend": backend,
            "profile_id": profile_id,
            "kb_name": kb_name,
            "num_sessions": 0,
            "transcript_path": "",
            "error": "0 entries",
        }

    transcript_path = output_root / "transcripts" / kb_name / backend / f"{profile_id}.json"
    existing = None if force else _load_existing_transcript(transcript_path)
    existing_sessions_by_entry: dict[str, dict] = {}
    if existing:
        for s in existing.get("sessions", []) or []:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("entry_id", "")).strip()
            if sid:
                existing_sessions_by_entry[sid] = s

    workspace = str(output_root / "workspaces" / kb_name / backend / profile_id)
    prior_sessions_summary: list[str] = []
    current_profile = entries[0].get("profile", {})
    sessions_results: list[dict] = []
    skipped_existing = 0
    newly_run = 0

    for i, base_entry in enumerate(entries):
        session_num = i + 1
        entry = dict(base_entry)
        entry_id = str(entry.get("entry_id", f"session_{session_num}"))

        if evolve_profile and i > 0:
            prev_profile = entries[i - 1].get("profile", {})
            resolved = entries[i - 1].get("gaps", [])
            current_profile = evolve_profile_fn(prev_profile, resolved)
        entry["profile"] = current_profile

        if entry_id in existing_sessions_by_entry:
            prev = existing_sessions_by_entry[entry_id]
            prev_transcript = prev.get("transcript", []) or []
            prev_entry = prev.get("entry", entry)
            result = {
                "entry_id": entry_id,
                "actual_turns": prev.get("actual_turns", 0),
                "transcript": prev_transcript,
                "entry": prev_entry,
                "practice_questions": prev.get("practice_questions", []) or [],
                "_skipped_existing": True,
            }
            summary = _summarize_session(
                result.get("transcript", []),
                result.get("entry", {}).get("task", {}),
                session_num,
            )
            prior_sessions_summary.append(summary)
            sessions_results.append(result)
            skipped_existing += 1
            logger.info(
                "[%s/%s] %s session %d/%d entry_id=%s already exists, skipped",
                kb_name,
                backend,
                profile_id,
                session_num,
                len(entries),
                entry_id,
            )
            continue

        prior_ctx = "\n".join(prior_sessions_summary) if prior_sessions_summary else None
        logger.info(
            "[%s/%s] %s session %d/%d",
            kb_name,
            backend,
            profile_id,
            session_num,
            len(entries),
        )

        try:
            session_buf = io.StringIO()
            with contextlib.redirect_stdout(session_buf):
                result = await _run_single_session(
                    entry=entry,
                    max_turns=max_turns,
                    auto=True,
                    use_editor=False,
                    auto_backend=backend,
                    deeptutor_workspace=workspace,
                    deeptutor_language=language,
                    prior_sessions_summary=prior_ctx,
                )
            if verbose:
                await _emit_dialogue_log(
                    kb_name=kb_name,
                    backend=backend,
                    profile_id=profile_id,
                    session_num=session_num,
                    session_total=len(entries),
                    text=session_buf.getvalue(),
                )
        except Exception as e:
            logger.error(
                "[%s/%s] %s session %d failed: %s",
                kb_name,
                backend,
                profile_id,
                session_num,
                e,
            )
            result = {
                "entry_id": entry.get("entry_id", f"session_{session_num}"),
                "transcript": [],
                "entry": entry,
                "actual_turns": 0,
                "practice_questions": [],
                "error": str(e),
            }

        summary = _summarize_session(result.get("transcript", []), entry.get("task", {}), session_num)
        prior_sessions_summary.append(summary)
        sessions_results.append(result)
        newly_run += 1

    combined = {
        "kb_name": kb_name,
        "profile_id": profile_id,
        "backend": backend,
        "timestamp": datetime.now().isoformat(),
        "mode": "auto",
        "evolve_profile": evolve_profile,
        "num_sessions": len(sessions_results),
        "sessions": [
            {
                "entry_id": r["entry_id"],
                "actual_turns": r["actual_turns"],
                "transcript": r.get("transcript", []),
                "entry": r["entry"],
                "practice_questions": r.get("practice_questions", []),
            }
            for r in sessions_results
        ],
    }

    transcript_path = output_root / "transcripts" / kb_name / backend / f"{profile_id}.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "backend": backend,
        "profile_id": profile_id,
        "kb_name": kb_name,
        "num_sessions": combined.get("num_sessions", 0),
        "num_skipped_existing": skipped_existing,
        "num_newly_run": newly_run,
        "transcript_path": str(transcript_path),
        "error": None,
    }


async def _process_profile(
    *,
    kb_name: str,
    profile_id: str,
    entries: list[dict],
    backends: list[str],
    output_root: Path,
    semaphore: asyncio.Semaphore,
    backend_semaphore: asyncio.Semaphore,
    max_turns: int,
    language: str,
    evolve_profile: bool,
    verbose: bool,
    force: bool,
) -> dict:
    async with semaphore:
        record = {
            "kb_name": kb_name,
            "profile_id": profile_id,
            "status": "ok",
            "num_entries": len(entries),
            "backends": {},
            "error": None,
        }
        async def _run_one_backend(backend: str) -> dict:
            async with backend_semaphore:
                return await _simulate_profile_backend(
                    kb_name=kb_name,
                    profile_id=profile_id,
                    entries=entries,
                    backend=backend,
                    output_root=output_root,
                    max_turns=max_turns,
                    language=language,
                    evolve_profile=evolve_profile,
                    verbose=verbose,
                    force=force,
                )

        async def _run_one_backend_tagged(backend: str):
            try:
                return backend, await _run_one_backend(backend), None
            except Exception as e:
                return backend, None, e

        backend_tasks = [
            asyncio.create_task(
                _run_one_backend_tagged(backend),
                name=f"{kb_name}:{profile_id}:{backend}",
            )
            for backend in backends
        ]
        total_backends = len(backend_tasks)
        completed_backends = 0
        for task in asyncio.as_completed(backend_tasks):
            completed_backends += 1
            pct = round(100.0 * completed_backends / total_backends, 1) if total_backends else 100.0
            backend, result, err = await task
            if err is not None:
                record["status"] = "error"
                record["backends"][backend] = {
                    "status": "error",
                    "error": str(err),
                }
                logger.error("[%s] %s backend=%s failed: %s", kb_name, profile_id, backend, err)
                continue

            record["backends"][backend] = result
            logger.info(
                "[%s] %s backend progress: %d/%d (%.1f%%) finished=%s status=%s",
                kb_name,
                profile_id,
                completed_backends,
                total_backends,
                pct,
                backend,
                result.get("status", "unknown"),
            )
        return record


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Step2: run simulations to generate transcripts")
    parser.add_argument("--kb-names", required=True, help="Comma-separated KB names to process")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Pipeline output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--backends",
        default="mock,deep_tutor",
        help=(
            "Comma-separated backends (parallel per profile). "
            "Supported: mock, deep_tutor, deep_tutor_no_rag, "
            "deep_tutor_no_memory, deep_tutor_no_rag_memory"
        ),
    )
    parser.add_argument("--concurrency", type=int, default=6, help="Max parallel profiles")
    parser.add_argument(
        "--backend-concurrency",
        type=int,
        default=4,
        help="Max parallel backends per profile (default: 4)",
    )
    parser.add_argument("--max-turns", type=int, default=30, help="Max student turns per session")
    parser.add_argument("--language", default="en", help="DeepTutor language")
    parser.add_argument(
        "--model",
        default="",
        help="Override LLM model for step2 simulation. If set, ignores env LLM_MODEL.",
    )
    parser.add_argument("--no-evolve", action="store_true", help="Disable profile evolution")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-turn student/tutor dialogue logs (newly executed sessions only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun all entries even if transcript already contains them.",
    )
    args = parser.parse_args()

    if args.model:
        # Force model override for this process.
        os.environ["LLM_MODEL"] = args.model
        try:
            from src.services.llm.config import clear_llm_config_cache

            clear_llm_config_cache()
        except Exception:
            pass

    # Suppress noisy logs from RAG/LLM internals during auto simulation
    from benchmark.simulation.conversation import _suppress_noisy_auto_logs

    _suppress_noisy_auto_logs()
    logging.getLogger("benchmark.pipeline.step2").setLevel(logging.INFO)
    logging.getLogger("benchmark.conversation").setLevel(logging.INFO)

    kb_names = _parse_names(args.kb_names)
    backends = _parse_names(args.backends)
    invalid_backends = [b for b in backends if b not in SUPPORTED_BACKENDS]
    if invalid_backends:
        raise ValueError(
            f"Unsupported backends: {invalid_backends}. "
            f"Supported: {sorted(SUPPORTED_BACKENDS)}"
        )
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    entries_root = output_root / "entries"
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    print(f"KBs: {len(kb_names)} | Concurrency(profile): {args.concurrency}")
    print(f"Backends(parallel/profile): {backends}")
    print(f"Backend concurrency/profile: {args.backend_concurrency}")
    print(f"Dialogue logs: {'enabled (-v/--verbose)' if args.verbose else 'disabled'}")
    print(f"Resume mode: {'disabled (--force)' if args.force else 'enabled (skip existing entries)'}")
    if args.verbose and not args.force:
        print("Note: in resume mode, existing sessions are skipped; dialogue logs are shown only for newly executed sessions.")
    print(f"Output root: {output_root}")

    sem = asyncio.Semaphore(args.concurrency)
    backend_sem = asyncio.Semaphore(max(1, args.backend_concurrency))
    tasks: list[asyncio.Task] = []
    pre_errors: list[dict] = []
    for kb_name in kb_names:
        kb_profiles_root = entries_root / kb_name / "profiles"
        if not kb_profiles_root.exists():
            pre_errors.append(
                {
                    "kb_name": kb_name,
                    "profile_id": None,
                    "status": "error",
                    "error": f"Missing entries directory: {kb_profiles_root}",
                }
            )
            logger.error("[%s] missing entries directory: %s", kb_name, kb_profiles_root)
            continue
        for profile_dir in sorted(p for p in kb_profiles_root.iterdir() if p.is_dir()):
            profile_id = profile_dir.name
            entries_path = profile_dir / "entries.jsonl"
            if not entries_path.exists():
                pre_errors.append(
                    {
                        "kb_name": kb_name,
                        "profile_id": profile_id,
                        "status": "error",
                        "error": f"Missing entries.jsonl: {entries_path}",
                    }
                )
                logger.error("[%s] %s missing entries.jsonl", kb_name, profile_id)
                continue
            try:
                entries = _load_entries(entries_path)
            except Exception as e:
                pre_errors.append(
                    {
                        "kb_name": kb_name,
                        "profile_id": profile_id,
                        "status": "error",
                        "error": f"Failed to load entries.jsonl: {e}",
                    }
                )
                continue
            if not entries:
                pre_errors.append(
                    {
                        "kb_name": kb_name,
                        "profile_id": profile_id,
                        "status": "error",
                        "error": f"entries.jsonl is empty (0 entries): {entries_path}",
                    }
                )
                logger.error("[%s] %s has 0 entries, skipping", kb_name, profile_id)
                continue
            async def _run_profile_tagged(kb_name: str, profile_id: str, entries: list[dict]):
                try:
                    return await _process_profile(
                    kb_name=kb_name,
                    profile_id=profile_id,
                    entries=entries,
                    backends=backends,
                    output_root=output_root,
                    semaphore=sem,
                    backend_semaphore=backend_sem,
                    max_turns=args.max_turns,
                    language=args.language,
                    evolve_profile=not args.no_evolve,
                    verbose=args.verbose,
                    force=args.force,
                    )
                except Exception as e:
                    return {
                        "kb_name": kb_name,
                        "profile_id": profile_id,
                        "status": "error",
                        "error": str(e),
                    }

            tasks.append(
                asyncio.create_task(
                    _run_profile_tagged(kb_name, profile_id, entries),
                    name=f"{kb_name}:{profile_id}",
                )
            )

    logger.info("Launching %d profile simulation tasks", len(tasks))
    profile_results = []
    task_errors = 0
    total_profiles = len(tasks)
    completed_profiles = 0
    for task in asyncio.as_completed(tasks):
        completed_profiles += 1
        pct = round(100.0 * completed_profiles / total_profiles, 1) if total_profiles else 100.0
        result = await task
        profile_results.append(result)
        if result.get("status") != "ok":
            task_errors += 1
        logger.info(
            "Profile progress: %d/%d (%.1f%%) finished=%s/%s status=%s",
            completed_profiles,
            total_profiles,
            pct,
            result.get("kb_name", "?"),
            result.get("profile_id", "?"),
            result.get("status", "unknown"),
        )

    manifest = {
        "step": "step2_generate_transcripts",
        "timestamp": datetime.now().isoformat(),
        "kb_names": kb_names,
        "backends": backends,
        "model": (args.model or os.getenv("LLM_MODEL", "")),
        "output_root": str(output_root),
        "concurrency_profile": args.concurrency,
        "backend_concurrency_per_profile": max(1, args.backend_concurrency),
        "backend_execution": "parallel_per_profile",
        "overwrite": bool(args.force),
        "pre_errors": pre_errors,
        "results": profile_results,
        "num_profiles": len(profile_results),
        "num_errors": len(pre_errors) + task_errors,
    }
    manifest_path = manifests_root / "step2_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nManifest: {manifest_path}")
    print(f"Done. Profile tasks: {len(profile_results)} | Errors: {manifest['num_errors']}")
    if manifest["num_errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
