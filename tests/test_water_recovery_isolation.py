"""Offline checks for the water-recovery isolation helpers.

Does not start Minecraft. Live evidence is
``obsidianlink/experiments/run_water_recovery_isolation.py``.
"""

from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.minedojo import MineDojoEnvironment
from obsidianlink.experiments.run_water_recovery_isolation import (
    analyze_window,
    consecutive_water_bucket_run,
    inventory_stable_tick,
)


_NO_OP = {
    "attack": 0,
    "back": 0,
    "camera": [0.0, 0.0],
    "forward": 0,
    "jump": 0,
    "left": 0,
    "right": 0,
    "sneak": 0,
    "use": 0,
    "equip": "none",
}


def test_wait_maps_use_zero_on_minedojo_keys() -> None:
    mapped = MineDojoEnvironment._to_minedojo_action(Action(type=ActionType.WAIT), _NO_OP)
    assert mapped["use"] == 0
    assert mapped["attack"] == 0
    assert mapped["forward"] == 0
    assert mapped["camera"] == [0.0, 0.0]


def test_consecutive_water_bucket_run_pins_appear_and_drop() -> None:
    trace = [
        {"tick": 10, "water_bucket": 0},
        {"tick": 11, "water_bucket": 1},
        {"tick": 12, "water_bucket": 1},
        {"tick": 13, "water_bucket": 1},
        {"tick": 14, "water_bucket": 0},
        {"tick": 15, "water_bucket": 0},
    ]
    run = consecutive_water_bucket_run(trace)
    assert run["water_bucket_first_tick"] == 11
    assert run["water_bucket_last_present_tick"] == 13
    assert run["water_bucket_duration_ticks"] == 3
    assert run["water_bucket_disappear_tick"] == 14
    assert run["rollback"] is True
    assert run["stable_at_end"] is False


def test_analyze_window_flags_wait_only_rollback() -> None:
    trace = [
        {
            "tick": 0,
            "action": "wait",
            "water_bucket": 1,
            "inventory": {"water_bucket": 1},
            "selected_item": "water_bucket",
            "reward": 0.0,
            "done": False,
            "pose": {"pitch": 58.0},
            "minerl": {"use": 0},
            "step_latency": 0.05,
        },
        {
            "tick": 1,
            "action": "wait",
            "water_bucket": 1,
            "inventory": {"water_bucket": 1},
            "selected_item": "water_bucket",
            "reward": 0.0,
            "done": False,
            "pose": {"pitch": 58.0},
            "minerl": {"use": 0},
            "step_latency": 0.05,
        },
        {
            "tick": 2,
            "action": "wait",
            "water_bucket": 0,
            "inventory": {"bucket": 2},
            "selected_item": "bucket",
            "reward": 0.0,
            "done": False,
            "pose": {"pitch": 58.0},
            "minerl": {"use": 0},
            "step_latency": 0.05,
        },
    ]
    analysis = analyze_window(trace)
    assert analysis["any_use"] is False
    assert analysis["any_forbidden"] is False
    assert analysis["minerl_use_max"] == 0
    assert analysis["water_bucket_disappear_tick"] == 2
    assert analysis["at_disappear"]["minerl_use"] == 0
    assert analysis["selected_item_changed"] is True
    assert analysis["reward_changed"] is False
    assert analysis["done_changed"] is False
    assert analysis["pose_changed"] is False


def test_inventory_stable_tick_is_last_change() -> None:
    trace = [
        {"tick": 0, "inventory": {"bucket": 1, "water_bucket": 1}},
        {"tick": 1, "inventory": {"bucket": 2}},
        {"tick": 2, "inventory": {"bucket": 2}},
        {"tick": 3, "inventory": {"bucket": 2}},
    ]
    assert inventory_stable_tick(trace) == 1
