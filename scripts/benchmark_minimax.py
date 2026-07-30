#!/usr/bin/env python3
"""Offline visual benchmark for MiniMax-M3 with thinking=disabled.

Implements Phase 6.3. Hardcodes the round-1 fixture set
(1 positive + 4 negatives) * 3 repeats = 15 real API calls, and the
expanded-fixture set
(2 positives + 4 negatives) * 3 repeats = 18 real API calls. Records
per-request results and aggregate metrics, including the project's
existing cave-candidate gate (``is_cave_candidate``,
``resolve_cave_direction``, ``has_directional_stone_bounded_dark_opening_region``,
``resolve_dark_opening_direction``).

This script is deliberately narrow: it reuses request/parse helpers from
``scripts/minimax_smoke.py`` and the project's existing cave-candidate
gates from ``mc_agent.actions``. It never opens MineRL, never relaxes
the existing geometry, text, direction, or ESC rules, and never
re-implements the gate. The Agent main flow
(``mc_agent.agent``) is untouched.

Prompt variants are explicitly registered via ``PROMPT_CONFIGS``. The
baseline prompt comes unchanged from ``mc_agent.qwen._prompt(None)``.
The V2 prompt (``prompt_v2_cave_salience``) appends a single visual
cave-salience paragraph to the baseline; it does not change the action
schema, image, model, endpoint, temperature, top_p,
``max_completion_tokens``, ``thinking`` setting, or local parse logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Make the sibling ``scripts/minimax_smoke.py`` importable as a module.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from minimax_smoke import _data_url, _request, _response_text  # noqa: E402

from mc_agent.actions import (  # noqa: E402
    MacroAction,
    has_directional_stone_bounded_dark_opening_region,
    is_cave_candidate,
    parse_macro_action,
    resolve_cave_direction,
    resolve_dark_opening_direction,
)
from mc_agent.qwen import _prompt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_ENDPOINT = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
DEFAULT_THINKING = "disabled"
REPEATS = 3

# Fixed sample set, defined in Phase 6.3 spec.
POSITIVE_FIXTURE_CENTER = ROOT / "tests/fixtures/genuine_cave_entrance/entrance.png"
POSITIVE_FIXTURE_RIGHT = (
    ROOT / "tests/fixtures/genuine_cave_entrance/after_approach_right.png"
)
NEGATIVE_FIXTURES = (
    ROOT / "tests/fixtures/seed101_t0_dirt_terrace_false_positive.png",
    ROOT / "tests/fixtures/seed3_frame_veto_regression/tick-7000.png",
    ROOT / "tests/fixtures/seed3_frame_veto_regression/tick-7109.png",
    ROOT / "tests/fixtures/seed3_frame_veto_regression/tick-7410.png",
)
EXPECTED_POSITIVE_DIRECTION = "center"
EXPECTED_APPROACH_DIRECTION = "right"

# ---------------------------------------------------------------------------
# Preregistered prompt variants.
# ---------------------------------------------------------------------------
PROMPT_CONFIG_BASELINE = "baseline"
PROMPT_CONFIG_V2 = "prompt_v2_cave_salience"
DEFAULT_PROMPT_CONFIG = PROMPT_CONFIG_BASELINE

PROMPT_V2_CAVE_SALIENCE_SUFFIX = (
    " Final cave check before returning JSON: independently inspect the "
    "left, center, and right image thirds for cave evidence; do not skip "
    "this check just because the center route looks walkable. Set "
    "cave_visible=true only when a continuous dark recessed area is "
    "visibly surrounded by gray stone or rock and looks enterable. Set "
    "cave_visible=false for shadows, dark patches under trees, water "
    "surface, dirt walls or dirt pits, flat nighttime darkness, and small "
    "distant dark spots. When cave_visible=true, the reason must follow "
    "the exact pattern 'dark stone opening on the left|center|right' and "
    "must name the third where the dark opening itself sits, not the "
    "walking direction."
)


def list_prompt_configs() -> tuple[str, ...]:
    return (PROMPT_CONFIG_BASELINE, PROMPT_CONFIG_V2)


def build_prompt(prompt_config: str) -> str:
    if prompt_config == PROMPT_CONFIG_BASELINE:
        return _prompt(None)
    if prompt_config == PROMPT_CONFIG_V2:
        return _prompt(None) + PROMPT_V2_CAVE_SALIENCE_SUFFIX
    raise ValueError(f"unknown prompt config: {prompt_config!r}")


def _build_payload(
    *, model: str, image: Path, thinking: str, prompt_config: str
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(image)}},
                    {"type": "text", "text": build_prompt(prompt_config)},
                ],
            }
        ],
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_completion_tokens": 256,
        "thinking": {"type": thinking},
    }


def build_samples() -> list[dict[str, Any]]:
    """Return the round-1 fixture set: 1 positive (center) + 4 negatives.

    The expanded-fixture round uses ``build_samples_expanded()``.
    """
    samples: list[dict[str, Any]] = [
        {
            "label": "positive",
            "path": POSITIVE_FIXTURE_CENTER,
            "expected_cave_visible": True,
            "expected_direction": EXPECTED_POSITIVE_DIRECTION,
        }
    ]
    for path in NEGATIVE_FIXTURES:
        samples.append(
            {
                "label": "negative",
                "path": path,
                "expected_cave_visible": False,
                "expected_direction": None,
            }
        )
    return samples


def build_samples_expanded() -> list[dict[str, Any]]:
    """Return the expanded-fixture set: 2 positives (center + right) + 4 negatives.

    The second positive is a post-approach view of the same genuine
    seed-101 entrance. The expected direction is "right" because the
    opening is offset to the right of the current view at that tick;
    see ``tests/fixtures/genuine_cave_entrance/README.md`` for the
    provenance record.
    """
    samples: list[dict[str, Any]] = [
        {
            "label": "positive",
            "path": POSITIVE_FIXTURE_CENTER,
            "expected_cave_visible": True,
            "expected_direction": EXPECTED_POSITIVE_DIRECTION,
        },
        {
            "label": "positive",
            "path": POSITIVE_FIXTURE_RIGHT,
            "expected_cave_visible": True,
            "expected_direction": EXPECTED_APPROACH_DIRECTION,
        },
    ]
    for path in NEGATIVE_FIXTURES:
        samples.append(
            {
                "label": "negative",
                "path": path,
                "expected_cave_visible": False,
                "expected_direction": None,
            }
        )
    return samples


def load_fixture_frame(image: Path) -> np.ndarray:
    """Load a fixture image as an RGB uint8 numpy array, exactly the shape
    the cave-candidate geometry gates expect (``(360, 640, 3)``).
    """
    return np.array(Image.open(image).convert("RGB"))


def evaluate_candidate_gate(
    frame: np.ndarray, action: MacroAction
) -> dict[str, Any]:
    """Run the project's full cave-candidate gate pipeline on a fixture frame.

    Mirrors the gate sequence in ``mc_agent.agent``:
      1. ``is_cave_candidate(action)`` — text evidence complete.
      2. ``resolve_cave_direction(reason)`` — claimed direction or None.
      3. ``has_directional_stone_bounded_dark_opening_region(frame, direction)``
         — geometry gate on the exact frame, restricted to the claimed band.
      4. Fallback: ``resolve_dark_opening_direction(frame)`` — local
         conservative direction from the actual frame, then re-test
         the geometry gate on the local direction.

    The script never relaxes the gate, the geometry rules, the text
    rules, the direction rules, or the ESC rules. It only records the
    outcome of the existing pipeline.

    Returns a dict with:
      - ``candidate_gate_passed``: bool
      - ``candidate_gate_direction``: final direction (model-claimed or
        local-resolved) that drove the gate, or None
      - ``candidate_direction_source``: ``"model_reason"``,
        ``"local_dark_region"``, or None
      - ``candidate_gate_failure_reason``: one of
        ``"cave_visible_false"``, ``"text_evidence_incomplete"``,
        ``"direction_unresolved"``, ``"geometry_veto"``, or None
    """
    if not action.cave_visible:
        return {
            "candidate_gate_passed": False,
            "candidate_gate_direction": None,
            "candidate_direction_source": None,
            "candidate_gate_failure_reason": "cave_visible_false",
        }
    if not is_cave_candidate(action):
        return {
            "candidate_gate_passed": False,
            "candidate_gate_direction": None,
            "candidate_direction_source": None,
            "candidate_gate_failure_reason": "text_evidence_incomplete",
        }
    direction = resolve_cave_direction(action.reason)
    if direction is None:
        return {
            "candidate_gate_passed": False,
            "candidate_gate_direction": None,
            "candidate_direction_source": None,
            "candidate_gate_failure_reason": "direction_unresolved",
        }
    if has_directional_stone_bounded_dark_opening_region(frame, direction):
        return {
            "candidate_gate_passed": True,
            "candidate_gate_direction": direction,
            "candidate_direction_source": "model_reason",
            "candidate_gate_failure_reason": None,
        }
    # Fallback: try the local conservative direction from the actual frame.
    local_dir = resolve_dark_opening_direction(frame)
    if local_dir is not None and has_directional_stone_bounded_dark_opening_region(
        frame, local_dir
    ):
        return {
            "candidate_gate_passed": True,
            "candidate_gate_direction": local_dir,
            "candidate_direction_source": "local_dark_region",
            "candidate_gate_failure_reason": None,
        }
    return {
        "candidate_gate_passed": False,
        "candidate_gate_direction": direction,
        "candidate_direction_source": "model_reason",
        "candidate_gate_failure_reason": "geometry_veto",
    }


def call_once(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    image: Path,
    thinking: str,
    timeout: float,
    prompt_config: str,
) -> dict[str, Any]:
    """Run one request and return a redacted result record. Never logs the key."""
    body = _build_payload(
        model=model, image=image, thinking=thinking, prompt_config=prompt_config,
    )
    started = time.perf_counter()
    record: dict[str, Any] = {
        "image": str(image.relative_to(ROOT)),
        "prompt_config": prompt_config,
        "latency_seconds": None,
        "request_id": None,
        "usage": None,
        "raw_response": None,
        "parser_accepted": None,
        "parser_error": None,
        "action": None,
        "candidate_gate_passed": None,
        "candidate_gate_direction": None,
        "candidate_direction_source": None,
        "candidate_gate_failure_reason": None,
        "error_type": None,
        "error": None,
    }
    try:
        response = _request(
            endpoint=endpoint,
            api_key=api_key,
            body=body,
            timeout=timeout,
        )
        record["latency_seconds"] = round(time.perf_counter() - started, 3)
        record["request_id"] = response.get("id")
        record["usage"] = response.get("usage")
        raw = _response_text(response)
        record["raw_response"] = raw
        parsed = parse_macro_action(raw)
        record["parser_accepted"] = parsed.accepted
        record["parser_error"] = parsed.error
        macro = parsed.action
        record["action"] = macro.to_log_dict()
        # Run the project's cave-candidate gate on the actual fixture frame
        # the model "saw" in this request. The frame is loaded from the
        # exact same path the model received.
        frame = load_fixture_frame(image)
        gate = evaluate_candidate_gate(frame, macro)
        record["candidate_gate_passed"] = gate["candidate_gate_passed"]
        record["candidate_gate_direction"] = gate["candidate_gate_direction"]
        record["candidate_direction_source"] = gate["candidate_direction_source"]
        record["candidate_gate_failure_reason"] = gate["candidate_gate_failure_reason"]
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        record["latency_seconds"] = round(time.perf_counter() - started, 3)
        record["error_type"] = type(error).__name__
        record["error"] = str(error)[:500]
    return record


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(float(sorted_values[0]), 3)
    k = (len(sorted_values) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(float(sorted_values[int(k)]), 3)
    lower = float(sorted_values[int(f)]) * (c - k)
    upper = float(sorted_values[int(c)]) * (k - f)
    return round(lower + upper, 3)


def _cave_visible(r: dict[str, Any]) -> bool | None:
    action = r.get("action")
    if not isinstance(action, dict):
        return None
    return bool(action.get("cave_visible"))


def _direction_is_expected(r: dict[str, Any], expected: str | None) -> bool:
    if expected is None:
        return True
    action = r.get("action") or {}
    reason = (action.get("reason") or "").lower()
    return expected in reason


def aggregate_metrics(
    records: list[dict[str, Any]], *, prompt_config: str
) -> dict[str, Any]:
    """Compute the Phase 6.3 summary metrics from per-request records."""
    total = len(records)
    parsed_ok = sum(1 for r in records if r.get("parser_accepted") is True)
    parse_rate = parsed_ok / total if total else 0.0

    positives = [r for r in records if r.get("label") == "positive"]
    negatives = [r for r in records if r.get("label") == "negative"]

    positive_recall_count = sum(1 for r in positives if _cave_visible(r) is True)
    positive_recall = (
        positive_recall_count / len(positives) if positives else 0.0
    )

    pos_recalled = [r for r in positives if _cave_visible(r) is True]
    pos_direction_ok = sum(
        1 for r in pos_recalled
        if _direction_is_expected(r, r.get("expected_direction"))
    )
    positive_direction_rate = (
        pos_direction_ok / len(pos_recalled) if pos_recalled else 0.0
    )

    pos_correct_full = sum(
        1 for r in positives
        if _cave_visible(r) is True
        and _direction_is_expected(r, r.get("expected_direction"))
    )
    positive_correct_detection_direction_rate = (
        pos_correct_full / len(positives) if positives else 0.0
    )

    pos_candidate_gate_count = sum(
        1 for r in positives if r.get("candidate_gate_passed") is True
    )
    positive_candidate_gate_pass_rate = (
        pos_candidate_gate_count / len(positives) if positives else 0.0
    )

    negative_candidate_gate_count = sum(
        1 for r in negatives if r.get("candidate_gate_passed") is True
    )
    negative_candidate_gate_pass_rate = (
        negative_candidate_gate_count / len(negatives) if negatives else 0.0
    )

    negative_fp = sum(1 for r in negatives if _cave_visible(r) is True)
    negative_fp_rate = negative_fp / len(negatives) if negatives else 0.0

    latencies = sorted(
        float(r["latency_seconds"])
        for r in records
        if r.get("latency_seconds") is not None
    )
    p50 = _percentile(latencies, 50) if latencies else None
    p95 = _percentile(latencies, 95) if latencies else None

    total_tokens = 0
    for r in records:
        usage = r.get("usage") or {}
        tokens = usage.get("total_tokens")
        if isinstance(tokens, int):
            total_tokens += tokens

    return {
        "prompt_config": prompt_config,
        "total_requests": total,
        "positive_samples": len(positives),
        "negative_samples": len(negatives),
        "strict_parse_success_rate": round(parse_rate, 4),
        "positive_recall_count": positive_recall_count,
        "positive_recall_rate": round(positive_recall, 4),
        "positive_direction_correct_count": pos_direction_ok,
        "positive_direction_correct_rate": round(positive_direction_rate, 4),
        "positive_correct_detection_direction_count": pos_correct_full,
        "positive_correct_detection_direction_rate": round(
            positive_correct_detection_direction_rate, 4
        ),
        "positive_candidate_gate_pass_count": pos_candidate_gate_count,
        "positive_candidate_gate_pass_rate": round(
            positive_candidate_gate_pass_rate, 4
        ),
        "negative_false_positive_count": negative_fp,
        "negative_false_positive_rate": round(negative_fp_rate, 4),
        "negative_candidate_gate_false_pass_count": negative_candidate_gate_count,
        "negative_candidate_gate_false_pass_rate": round(
            negative_candidate_gate_pass_rate, 4
        ),
        "latency_p50_seconds": p50,
        "latency_p95_seconds": p95,
        "total_tokens": total_tokens,
        "thresholds": {
            "strict_parse_success_required": 1.0,
            "negative_false_positive_rate_max": 0.0,
            "negative_candidate_gate_false_pass_rate_max": 0.0,
            "positive_recall_rate_required": 1.0,
            "positive_correct_detection_direction_rate_required": 1.0,
            "positive_candidate_gate_pass_rate_required": 1.0,
        },
        "all_thresholds_met": (
            parse_rate == 1.0
            and negative_fp_rate == 0.0
            and negative_candidate_gate_pass_rate == 0.0
            and positive_recall == 1.0
            and positive_correct_detection_direction_rate == 1.0
            and positive_candidate_gate_pass_rate == 1.0
        ),
    }


def write_results_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summary(summary: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prompt_source_line(prompt_config: str) -> str:
    if prompt_config == PROMPT_CONFIG_BASELINE:
        return "unmodified `mc_agent.qwen._prompt(None)`"
    return (
        "`mc_agent.qwen._prompt(None)` with the preregistered "
        "`prompt_v2_cave_salience` suffix appended at the end of the "
        "user message. The suffix is restricted to observable visual "
        "description (see "
        "`scripts/benchmark_minimax.py::PROMPT_V2_CAVE_SALIENCE_SUFFIX`)."
    )


def write_readme(
    meta: dict[str, Any],
    *,
    fixture_set_label: str,
    fixture_set_description: str,
    path: Path,
) -> None:
    lines = [
        f"# Phase 6.3 — MiniMax-M3 offline visual benchmark — `{meta['prompt_config']}` ({fixture_set_label})",
        "",
        f"- Started: `{meta['started_at']}`",
        f"- Provider: `{meta['provider']}`",
        f"- Model: `{meta['model']}`",
        f"- Endpoint: `{meta['endpoint']}`",
        f"- Thinking: `{meta['thinking']}`",
        f"- Prompt config: `{meta['prompt_config']}`",
        f"- Prompt source: {_prompt_source_line(meta['prompt_config'])}",
        "- Image input format: `image_url` data URL (PNG/JPEG/WebP), same as `scripts/minimax_smoke.py`.",
        f"- Total API calls: **{meta['total_calls']}**.",
        "",
        "## Sample set",
        "",
        fixture_set_description,
        "",
        "## Pass criteria for Worker design",
        "",
        "- 100 % strict parser success.",
        "- 100 % negatives: `cave_visible=false` AND `candidate_gate_passed=false`.",
        "- 100 % positives: `cave_visible=true` AND reason contains the expected direction AND `candidate_gate_passed=true`.",
        "",
        "## Cave-candidate gate",
        "",
        "Every per-request record carries the project's full candidate-gate",
        "outcome: `candidate_gate_passed`, `candidate_gate_direction`,",
        "`candidate_direction_source` (`model_reason` | `local_dark_region` |",
        "None), and `candidate_gate_failure_reason` (one of",
        "`cave_visible_false`, `text_evidence_incomplete`,",
        "`direction_unresolved`, `geometry_veto`, or None). The gate uses",
        "the existing ``mc_agent.actions`` functions only and never relaxes",
        "the geometry, text, direction, or ESC rules.",
        "",
        "## Files",
        "",
        "- `results.jsonl` — per-request results, one JSON object per line.",
        "- `summary.json` — aggregate metrics and threshold check.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round1_fixture_description() -> str:
    return (
        "Positives (1):\n"
        "- `tests/fixtures/genuine_cave_entrance/entrance.png` — verified "
        "byte-for-byte copy of `runs/manual-findcave/20260723-110256/candidates/candidate-tick-03026.png` "
        "(MD5 `a5cf9195457f25a17d1bc527f4b08651`, 144431 bytes). See "
        "`tests/fixtures/genuine_cave_entrance/README.md` for the full "
        "provenance record. Expected direction: `center`.\n"
        "\n"
        "Negatives (4):\n"
        "- `tests/fixtures/seed101_t0_dirt_terrace_false_positive.png` — dirt terrace (known Qwen false positive).\n"
        "- `tests/fixtures/seed3_frame_veto_regression/tick-7000.png` — Phase 4 veto regression frame.\n"
        "- `tests/fixtures/seed3_frame_veto_regression/tick-7109.png` — Phase 4 veto regression frame.\n"
        "- `tests/fixtures/seed3_frame_veto_regression/tick-7410.png` — Phase 4 veto regression frame."
    )


def _expanded_fixture_description() -> str:
    return (
        "Positives (2):\n"
        "- `tests/fixtures/genuine_cave_entrance/entrance.png` — verified "
        "byte-for-byte copy of "
        "`runs/manual-findcave/20260723-110256/candidates/candidate-tick-03026.png` "
        "(MD5 `a5cf9195457f25a17d1bc527f4b08651`, 144431 bytes). Expected "
        "direction: `center`. See "
        "`tests/fixtures/genuine_cave_entrance/README.md` for the full "
        "provenance record.\n"
        "- `tests/fixtures/genuine_cave_entrance/after_approach_right.png` — "
        "verified byte-for-byte copy of "
        "`runs/phase4-true-entrance-approach/20260723-142315/episode-01/decision_frames/tick-0235.png` "
        "(MD5 `8d814f039bfb2983a5e8c1022a04b559`, SHA-256 "
        "`e1d4e1318d06be6ba82ea1dbeb40fe000d1e8e282ee97b332437f75a5627d6e9`, "
        "207870 bytes). Manual validation: "
        "`runs/phase4-true-entrance-approach/20260723-142315/episode-01/manual_review.md`. "
        "The same stone-bounded entrance after approach, offset to the "
        "right of the current view. Expected direction: `right`.\n"
        "\n"
        "Negatives (4):\n"
        "- `tests/fixtures/seed101_t0_dirt_terrace_false_positive.png` — dirt terrace (known Qwen false positive).\n"
        "- `tests/fixtures/seed3_frame_veto_regression/tick-7000.png` — Phase 4 veto regression frame.\n"
        "- `tests/fixtures/seed3_frame_veto_regression/tick-7109.png` — Phase 4 veto regression frame.\n"
        "- `tests/fixtures/seed3_frame_veto_regression/tick-7410.png` — Phase 4 veto regression frame."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory; default runs/phase6-minimax-benchmark/<ts>/",
    )
    parser.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MINIMAX_ENDPOINT", DEFAULT_ENDPOINT),
    )
    parser.add_argument(
        "--thinking",
        choices=("disabled", "adaptive", "enabled"),
        default=DEFAULT_THINKING,
    )
    parser.add_argument(
        "--prompt-config",
        choices=list_prompt_configs(),
        default=DEFAULT_PROMPT_CONFIG,
        help="Preregistered prompt variant. Default: baseline (unchanged Qwen prompt).",
    )
    parser.add_argument(
        "--fixture-set",
        choices=("round1", "expanded"),
        default="round1",
        help=(
            "Sample-set to use. round1 = 1 positive + 4 negatives (15 calls). "
            "expanded = 2 positives + 4 negatives (18 calls). "
            "The expanded set adds the after-approach right-direction positive."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        parser.error("MINIMAX_API_KEY must be set in the environment")

    if args.fixture_set == "expanded":
        samples = build_samples_expanded()
        fixture_set_label = "expanded-fixtures"
        fixture_set_description = _expanded_fixture_description()
        output_suffix = "expanded-fixtures"
    else:
        samples = build_samples()
        fixture_set_label = "round1"
        fixture_set_description = _round1_fixture_description()
        output_suffix = ""

    missing = [str(s["path"]) for s in samples if not s["path"].is_file()]
    if missing:
        parser.error(f"missing fixtures: {missing}")

    if args.output_dir is not None:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prompt_tag = (
            f"-{args.prompt_config}" if args.prompt_config != PROMPT_CONFIG_BASELINE else ""
        )
        # round1 defaults leave the directory flat to preserve the
        # round-1 naming convention. The expanded-fixture round always
        # tags its directory so the two rounds stay distinguishable.
        if output_suffix and prompt_tag:
            default_name = f"{stamp}{prompt_tag}-{output_suffix}"
        elif output_suffix:
            default_name = f"{stamp}-{output_suffix}"
        elif prompt_tag:
            default_name = f"{stamp}{prompt_tag}"
        else:
            default_name = stamp
        output_dir = ROOT / "runs" / "phase6-minimax-benchmark" / default_name
        output_dir.mkdir(parents=True, exist_ok=False)

    started_at = datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    for sample in samples:
        for repeat in range(1, REPEATS + 1):
            record = call_once(
                endpoint=args.endpoint,
                api_key=api_key,
                model=args.model,
                image=sample["path"],
                thinking=args.thinking,
                timeout=args.timeout_seconds,
                prompt_config=args.prompt_config,
            )
            record["repeat"] = repeat
            record["label"] = sample["label"]
            record["expected_cave_visible"] = sample["expected_cave_visible"]
            record["expected_direction"] = sample.get("expected_direction")
            records.append(record)
            cave_visible = (
                record["action"].get("cave_visible") if record["action"] else None
            )
            print(
                f"[bench] {record['image']} repeat {repeat} "
                f"-> latency={record['latency_seconds']}s "
                f"parser_accepted={record['parser_accepted']} "
                f"cave_visible={cave_visible} "
                f"gate={record['candidate_gate_passed']} "
                f"reason={record['candidate_gate_failure_reason']} "
                f"error={record['error_type']}"
            )

    summary = aggregate_metrics(records, prompt_config=args.prompt_config)
    write_results_jsonl(records, output_dir / "results.jsonl")
    write_summary(summary, output_dir / "summary.json")
    write_readme(
        meta={
            "started_at": started_at,
            "provider": "minimax",
            "model": args.model,
            "endpoint": args.endpoint,
            "thinking": args.thinking,
            "prompt_config": args.prompt_config,
            "total_calls": len(records),
        },
        fixture_set_label=fixture_set_label,
        fixture_set_description=fixture_set_description,
        path=output_dir / "README.md",
    )
    print(f"[bench] wrote {output_dir}/results.jsonl")
    print(f"[bench] wrote {output_dir}/summary.json")
    print(f"[bench] wrote {output_dir}/README.md")
    print(f"[bench] summary: {json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
