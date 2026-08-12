from enum import Enum


class PortalConstructionLevel(str, Enum):
    L1_CONTROLLED_CONSTRUCTION = "L1"
    L2_RESOURCE_INTERACTION = "L2"
    L3_RESOURCE_ACQUISITION = "L3"
    L4_OPEN_WORLD_CONSTRUCTION = "L4"
