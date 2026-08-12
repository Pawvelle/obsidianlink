"""Coordination condition vocabulary only; no gameplay strategy."""

from enum import Enum


class MultiAgentComparison(str, Enum):
    NATURAL = "natural_multi_agent"
    COMPUTE_MATCHED = "compute_matched_multi_agent"
