"""Research conditions and natural portal-construction roles."""

from enum import Enum


class RoleAssignmentMode(str, Enum):
    FIXED_ROLE = "fixed_role"
    AUTONOMOUS_ROLE_ASSIGNMENT = "autonomous_role_assignment"


class TeamRole(str, Enum):
    LAVA_SCOUT = "lava_scout"
    MINER_CRAFTER = "miner_crafter"
    WATER_SCOUT = "water_scout"
    PORTAL_ASSEMBLER = "portal_assembler"
