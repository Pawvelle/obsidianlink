from enum import Enum


class DiagnosticLevel(str, Enum):
    D1_PERCEPTION = "D1"
    D2_GROUNDING = "D2"
    D3_MANIPULATION = "D3"
    D4_PLANNING = "D4"
    D5_STATE_TRACKING = "D5"
    D6_RECOVERY = "D6"
