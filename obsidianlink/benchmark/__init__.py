"""Solver-independent ObsidianLink v2 benchmark kernel contracts."""

from obsidianlink.benchmark.evaluator import Evaluator, EvaluatorVerdict
from obsidianlink.benchmark.evidence import (
    EvidenceChannel,
    EvidenceIdentity,
    EvidenceRecord,
    VerificationLevel,
)
from obsidianlink.benchmark.metrics import MetricName, MetricRecord
from obsidianlink.benchmark.runner import BenchmarkRunner, RunnerResult
from obsidianlink.benchmark.splits import BenchmarkSplit
from obsidianlink.benchmark.task import (
    BenchmarkSuite,
    ExecutionMode,
    LayoutType,
    TaskIdentity,
)

__all__ = [
    "BenchmarkRunner",
    "BenchmarkSplit",
    "BenchmarkSuite",
    "Evaluator",
    "EvaluatorVerdict",
    "EvidenceChannel",
    "EvidenceIdentity",
    "EvidenceRecord",
    "ExecutionMode",
    "LayoutType",
    "MetricName",
    "MetricRecord",
    "RunnerResult",
    "TaskIdentity",
    "VerificationLevel",
]
