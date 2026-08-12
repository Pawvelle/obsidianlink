"""v2 registry facade over the compatibility-preserving catalog parser."""

from obsidianlink.core.task_catalog import (
    ACTIVE_PHASE,
    TaskCatalog,
    TaskCatalogEntry,
    TaskTaxonomy,
    load_task_catalog,
    validate_catalog_references,
)

__all__ = [
    "ACTIVE_PHASE",
    "TaskCatalog",
    "TaskCatalogEntry",
    "TaskTaxonomy",
    "load_task_catalog",
    "validate_catalog_references",
]
