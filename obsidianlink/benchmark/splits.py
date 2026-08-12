"""Formal benchmark split vocabulary."""

from enum import Enum


class BenchmarkSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
