from enum import Enum


class PortalConstructionLevel(str, Enum):
    P1_CONTROLLED_CONSTRUCTION = "P1"
    P2_RESOURCE_INTERACTION = "P2"
    P3_RESOURCE_ACQUISITION = "P3"
    P4_OPEN_WORLD_CONSTRUCTION = "P4"
