"""Smallest P1 E0 recording helper.

Persists a deterministic JSON snapshot of an environment-validation
result. Recording never promotes verification level, never writes
benchmark evaluator verdicts, and never turns an offline result into
integration evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from obsidianlink.env.validation.result import (
    UNIT_VERIFIED,
    EnvironmentValidationResult,
)


class EnvironmentValidationRecorder:
    """Write E0 validation evidence without capability promotion."""

    def record(self, result: EnvironmentValidationResult, path: Path) -> Path:
        if not isinstance(result, EnvironmentValidationResult):
            raise ValueError("result must be EnvironmentValidationResult")
        if not isinstance(path, Path):
            raise ValueError("path must be a pathlib.Path")
        payload = result.as_dict()
        if payload["verification_level"] != UNIT_VERIFIED:
            raise ValueError("recorder refuses non-unit verification levels")
        if payload["integration_verified"] or payload["real_execution_performed"]:
            raise ValueError("recorder refuses integration or real-execution claims")
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        path.write_text(serialized + "\n", encoding="utf-8")
        return path
