"""P1 E0--E3 MineRL integration boundary.

This package may depend on the validation core and on the MineRL backend.
The solver-independent validation core must not import this package.
Importing these modules never starts MineRL or Minecraft.
"""

from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter
from obsidianlink.env.integration.e0_cleanup import E0CleanupStatus
from obsidianlink.env.integration.e0_run import (
    AUTHORIZED_LIVE_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E0,
    E0AuthorizationError,
    E0MineRLRunRecord,
    preflight_authorized_e0,
    run_authorized_e0_minerl,
)
from obsidianlink.env.integration.e1_adapter import MineRLE1RGBAdapter
from obsidianlink.env.integration.e1_run import (
    AUTHORIZED_LIVE_E1_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E1,
    E1AuthorizationError,
    E1MineRLRunRecord,
    preflight_authorized_e1,
    run_authorized_e1_minerl,
)
from obsidianlink.env.integration.e2_adapter import MineRLE2InventoryAdapter
from obsidianlink.env.integration.e2_run import (
    AUTHORIZED_LIVE_E2_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E2,
    E2AuthorizationError,
    E2MineRLRunRecord,
    preflight_authorized_e2,
    run_authorized_e2_minerl,
)
from obsidianlink.env.integration.e3_adapter import MineRLE3SelectedItemAdapter
from obsidianlink.env.integration.e3_run import (
    AUTHORIZED_LIVE_E3_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E3,
    E3AuthorizationError,
    E3MineRLRunRecord,
    preflight_authorized_e3,
    run_authorized_e3_minerl,
)

__all__ = [
    "AUTHORIZED_LIVE_E1_RUN_VALUE",
    "AUTHORIZED_LIVE_E2_RUN_VALUE",
    "AUTHORIZED_LIVE_E3_RUN_VALUE",
    "AUTHORIZED_LIVE_RUN_VALUE",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E0",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E1",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E2",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E3",
    "E0AuthorizationError",
    "E0CleanupStatus",
    "E0MineRLRunRecord",
    "E1AuthorizationError",
    "E1MineRLRunRecord",
    "E2AuthorizationError",
    "E2MineRLRunRecord",
    "E3AuthorizationError",
    "E3MineRLRunRecord",
    "MineRLE0LifecycleAdapter",
    "MineRLE1RGBAdapter",
    "MineRLE2InventoryAdapter",
    "MineRLE3SelectedItemAdapter",
    "preflight_authorized_e0",
    "preflight_authorized_e1",
    "preflight_authorized_e2",
    "preflight_authorized_e3",
    "run_authorized_e0_minerl",
    "run_authorized_e1_minerl",
    "run_authorized_e2_minerl",
    "run_authorized_e3_minerl",
]
