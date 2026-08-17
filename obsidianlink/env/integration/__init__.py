"""P1 E0--E12 MineRL integration boundary.

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
from obsidianlink.env.integration.e4_adapter import MineRLE4CameraAdapter
from obsidianlink.env.integration.e4_run import (
    AUTHORIZED_LIVE_E4_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E4,
    E4AuthorizationError,
    E4MineRLRunRecord,
    preflight_authorized_e4,
    run_authorized_e4_minerl,
)
from obsidianlink.env.integration.e5_adapter import MineRLE5MovementAdapter
from obsidianlink.env.integration.e5_run import (
    AUTHORIZED_LIVE_E5_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E5,
    E5AuthorizationError,
    E5MineRLRunRecord,
    preflight_authorized_e5,
    run_authorized_e5_minerl,
)
from obsidianlink.env.integration.e6_adapter import MineRLE6PlacementAdapter
from obsidianlink.env.integration.e6_run import (
    AUTHORIZED_LIVE_E6_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E6,
    E6AuthorizationError,
    E6MineRLRunRecord,
    preflight_authorized_e6,
    run_authorized_e6_minerl,
)
from obsidianlink.env.integration.e7_adapter import MineRLE7BucketAdapter
from obsidianlink.env.integration.e7_run import (
    AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE,
    AUTHORIZED_LIVE_E7_WATER_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E7_LAVA,
    EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER,
    E7AuthorizationError,
    E7MineRLRunRecord,
    preflight_authorized_e7,
    run_authorized_e7_minerl,
)
from obsidianlink.env.integration.e8_adapter import MineRLE8BlockTruthAdapter
from obsidianlink.env.integration.e8_run import (
    AUTHORIZED_LIVE_E8_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E8,
    E8AuthorizationError,
    E8MineRLRunRecord,
    preflight_authorized_e8,
    run_authorized_e8_minerl,
)
from obsidianlink.env.integration.e9_adapter import MineRLE9FluidTruthAdapter
from obsidianlink.env.integration.e9_run import (
    AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE,
    AUTHORIZED_LIVE_E9_WATER_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E9_LAVA,
    EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER,
    E9AuthorizationError,
    E9MineRLRunRecord,
    preflight_authorized_e9,
    run_authorized_e9_minerl,
)
from obsidianlink.env.integration.e10_adapter import MineRLE10ObsidianAdapter
from obsidianlink.env.integration.e10_geometry import (
    AUTHORIZED_LIVE_E10_GEOMETRY_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E10_GEOMETRY,
    preflight_authorized_e10_geometry,
    run_authorized_e10_geometry_smoke,
)
from obsidianlink.env.integration.e10_run import (
    AUTHORIZED_LIVE_E10_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E10,
    E10AuthorizationError,
    E10MineRLRunRecord,
    preflight_authorized_e10,
    run_authorized_e10_minerl,
)
from obsidianlink.env.integration.e11_adapter import MineRLE11PortalActivationAdapter
from obsidianlink.env.integration.e11_geometry import (
    AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY,
    preflight_authorized_e11_geometry,
    run_authorized_e11_geometry_smoke,
)
from obsidianlink.env.integration.e11_run import (
    AUTHORIZED_LIVE_E11_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E11,
    E11AuthorizationError,
    E11MineRLRunRecord,
    preflight_authorized_e11,
    run_authorized_e11_minerl,
)
from obsidianlink.env.integration.e12_adapter import MineRLE12DimensionTransitionAdapter
from obsidianlink.env.integration.e12_run import (
    AUTHORIZED_LIVE_E12_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E12,
    E12AuthorizationError,
    E12MineRLRunRecord,
    preflight_authorized_e12,
    run_authorized_e12_minerl,
)

__all__ = [
    "AUTHORIZED_LIVE_E1_RUN_VALUE",
    "AUTHORIZED_LIVE_E2_RUN_VALUE",
    "AUTHORIZED_LIVE_E3_RUN_VALUE",
    "AUTHORIZED_LIVE_E4_RUN_VALUE",
    "AUTHORIZED_LIVE_E5_RUN_VALUE",
    "AUTHORIZED_LIVE_E6_RUN_VALUE",
    "AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE",
    "AUTHORIZED_LIVE_E7_WATER_RUN_VALUE",
    "AUTHORIZED_LIVE_E8_RUN_VALUE",
    "AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE",
    "AUTHORIZED_LIVE_E9_WATER_RUN_VALUE",
    "AUTHORIZED_LIVE_E10_GEOMETRY_RUN_VALUE",
    "AUTHORIZED_LIVE_E10_RUN_VALUE",
    "AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE",
    "AUTHORIZED_LIVE_E11_RUN_VALUE",
    "AUTHORIZED_LIVE_E12_RUN_VALUE",
    "AUTHORIZED_LIVE_RUN_VALUE",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E0",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E1",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E2",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E3",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E4",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E5",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E6",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E7_LAVA",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E8",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E9_LAVA",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E10",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E10_GEOMETRY",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E11",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY",
    "EXECUTION_MODE_AUTHORIZED_LIVE_E12",
    "E0AuthorizationError",
    "E0CleanupStatus",
    "E0MineRLRunRecord",
    "E1AuthorizationError",
    "E1MineRLRunRecord",
    "E2AuthorizationError",
    "E2MineRLRunRecord",
    "E3AuthorizationError",
    "E3MineRLRunRecord",
    "E4AuthorizationError",
    "E4MineRLRunRecord",
    "E5AuthorizationError",
    "E5MineRLRunRecord",
    "E6AuthorizationError",
    "E6MineRLRunRecord",
    "E7AuthorizationError",
    "E7MineRLRunRecord",
    "E8AuthorizationError",
    "E8MineRLRunRecord",
    "E9AuthorizationError",
    "E9MineRLRunRecord",
    "E10AuthorizationError",
    "E10MineRLRunRecord",
    "E11AuthorizationError",
    "E11MineRLRunRecord",
    "E12AuthorizationError",
    "E12MineRLRunRecord",
    "MineRLE0LifecycleAdapter",
    "MineRLE1RGBAdapter",
    "MineRLE2InventoryAdapter",
    "MineRLE3SelectedItemAdapter",
    "MineRLE4CameraAdapter",
    "MineRLE5MovementAdapter",
    "MineRLE6PlacementAdapter",
    "MineRLE7BucketAdapter",
    "MineRLE8BlockTruthAdapter",
    "MineRLE9FluidTruthAdapter",
    "MineRLE10ObsidianAdapter",
    "MineRLE11PortalActivationAdapter",
    "MineRLE12DimensionTransitionAdapter",
    "preflight_authorized_e0",
    "preflight_authorized_e1",
    "preflight_authorized_e2",
    "preflight_authorized_e3",
    "preflight_authorized_e4",
    "preflight_authorized_e5",
    "preflight_authorized_e6",
    "preflight_authorized_e7",
    "preflight_authorized_e8",
    "preflight_authorized_e9",
    "preflight_authorized_e10",
    "preflight_authorized_e10_geometry",
    "preflight_authorized_e11",
    "preflight_authorized_e11_geometry",
    "preflight_authorized_e12",
    "run_authorized_e0_minerl",
    "run_authorized_e1_minerl",
    "run_authorized_e2_minerl",
    "run_authorized_e3_minerl",
    "run_authorized_e4_minerl",
    "run_authorized_e5_minerl",
    "run_authorized_e6_minerl",
    "run_authorized_e7_minerl",
    "run_authorized_e8_minerl",
    "run_authorized_e9_minerl",
    "run_authorized_e10_minerl",
    "run_authorized_e10_geometry_smoke",
    "run_authorized_e11_geometry_smoke",
    "run_authorized_e11_minerl",
    "run_authorized_e12_minerl",
]
