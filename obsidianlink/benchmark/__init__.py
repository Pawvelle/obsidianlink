"""Solver-independent ObsidianLink v2 benchmark kernel contracts."""

from obsidianlink.benchmark.evaluator import Evaluator, EvaluatorVerdict
from obsidianlink.benchmark.evidence import (
    EvidenceChannel,
    EvidenceIdentity,
    EvidenceRecord,
    VerificationLevel,
)
from obsidianlink.benchmark.metrics import MetricName, MetricRecord
from obsidianlink.benchmark.run_record import (
    BenchmarkRunRecord,
    load_run_record,
    run_benchmark,
    write_run_record,
)
from obsidianlink.benchmark.runner import BenchmarkRunner, RunnerResult
from obsidianlink.benchmark.splits import BenchmarkSplit
from obsidianlink.benchmark.task import (
    BenchmarkSuite,
    ExecutionMode,
    LayoutType,
    TaskIdentity,
)

__all__ = [
    "BenchmarkRunRecord",
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
    "load_run_record",
    "run_benchmark",
    "write_run_record",
]
