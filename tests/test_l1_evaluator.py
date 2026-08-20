"""Offline checks for the L1 Evaluator.

Does not start Minecraft. Never uses ``ObservationFromGrid``. Live
handler wiring is verified by
``obsidianlink/experiments/run_l1_evaluator_smoke.py``.
"""

from obsidianlink.benchmark.l1_evaluator import (
    NETHER_BIOME_ID,
    BiomeSample,
    L1Evaluator,
    L1Milestones,
    leaked_evaluator_fields,
    portal_activated_from_rewards,
    resolve_nether_entered,
)
from obsidianlink.env.environment import Observation
from obsidianlink.tasks.portal import L1_PORTAL_TASK


def test_portal_activated_requires_positive_reward_sample() -> None:
    assert portal_activated_from_rewards([0.0, 0.0, 1.0]) is True
    assert portal_activated_from_rewards([0.0, 0.0, None]) is False
    assert portal_activated_from_rewards([]) is False


def test_nether_entered_requires_activation_and_strict_biome_match() -> None:
    samples = [
        BiomeSample(reward=0.0, biome_id=1.0),
        BiomeSample(reward=1.0, biome_id=1.0),  # touched portal, still Overworld
        BiomeSample(reward=0.0, biome_id=NETHER_BIOME_ID),  # transitioned
    ]
    result = resolve_nether_entered(samples, baseline_biome_id=1.0)
    assert result["portal_activated"] is True
    assert result["nether_biome_strict_match"] is True
    assert result["nether_entered"] is True


def test_nether_entered_false_without_activation_even_if_biome_looks_nether() -> None:
    # Biome noise alone (e.g. a bad reading) must not be treated as success.
    samples = [BiomeSample(reward=0.0, biome_id=NETHER_BIOME_ID)]
    result = resolve_nether_entered(samples, baseline_biome_id=1.0)
    assert result["portal_activated"] is False
    assert result["nether_entered"] is False


def test_nether_entered_false_on_weak_biome_change_only() -> None:
    # Activated, biome changed, but not to the strict Nether id -> not success.
    samples = [
        BiomeSample(reward=0.0, biome_id=1.0),
        BiomeSample(reward=1.0, biome_id=1.0),
        BiomeSample(reward=0.0, biome_id=99.0),
    ]
    result = resolve_nether_entered(samples, baseline_biome_id=1.0)
    assert result["portal_activated"] is True
    assert result["nether_biome_strict_match"] is False
    assert result["biome_changed_weak"] is True
    assert result["nether_entered"] is False


def test_biome_change_before_activation_does_not_count() -> None:
    # Nether-id sample happens before the portal touch reward fires.
    samples = [
        BiomeSample(reward=0.0, biome_id=NETHER_BIOME_ID),
        BiomeSample(reward=1.0, biome_id=1.0),
    ]
    result = resolve_nether_entered(samples, baseline_biome_id=1.0)
    assert result["nether_entered"] is False


def test_milestones_accumulate_across_steps() -> None:
    m = L1Milestones()
    m.observe({"reward": 0.0, "biome_id": 1.0})
    m.observe({"reward": 1.0, "biome_id": 1.0})
    m.observe({"reward": 0.0, "biome_id": NETHER_BIOME_ID})
    m.observe(None)  # non-mapping hidden_state must not crash accumulation
    resolved = m.resolve()
    assert resolved["nether_entered"] is True
    assert m.baseline_biome_id == 1.0


def test_leaked_evaluator_fields_detects_extra_attributes() -> None:
    obs = Observation(frame=None, inventory={}, selected_item=None)
    assert leaked_evaluator_fields(obs) == []
    assert leaked_evaluator_fields(None) == []


def test_l1_evaluator_success_requires_nether_entry() -> None:
    ev = L1Evaluator()
    ev.observe_step({"reward": 0.0, "biome_id": 1.0})
    ev.observe_step({"reward": 1.0, "biome_id": 1.0})
    result = ev.evaluate(
        L1_PORTAL_TASK,
        steps=3,
        model_calls=0,
        invalid_actions=0,
        elapsed_time=1.0,
        hidden_state={"reward": 0.0, "biome_id": NETHER_BIOME_ID},
    )
    assert result.success is True
    assert result.evidence["milestones"]["nether_entered"] is True
    assert result.evidence["milestones"]["portal_constructed"] == "unknown"


def test_l1_evaluator_fails_closed_without_nether_entry() -> None:
    ev = L1Evaluator()
    ev.observe_step({"reward": 0.0, "biome_id": 1.0})
    result = ev.evaluate(
        L1_PORTAL_TASK,
        steps=1,
        model_calls=0,
        invalid_actions=0,
        elapsed_time=1.0,
        hidden_state={"reward": 0.0, "biome_id": 1.0},
    )
    assert result.success is False
    assert result.evidence["reason"] == "nether_entry_not_confirmed"


def test_l1_evaluator_never_reads_observation_from_grid() -> None:
    import inspect

    from obsidianlink.benchmark import l1_evaluator

    source = inspect.getsource(l1_evaluator)
    # Mentioning the banned handler in a comment/docstring is fine; calling
    # or indexing it as a truth source is not.
    assert "handlers.ObservationFromGrid" not in source
    assert '"l1_grid"' not in source
    assert "'l1_grid'" not in source


def test_l1_evaluator_evaluation_error_on_leaked_observation() -> None:
    class LeakyObservation:
        biome_id = 8

    ev = L1Evaluator()
    result = ev.evaluate(
        L1_PORTAL_TASK,
        steps=1,
        model_calls=0,
        invalid_actions=0,
        elapsed_time=1.0,
        observation=LeakyObservation(),
        hidden_state={"reward": 1.0, "biome_id": NETHER_BIOME_ID},
    )
    assert result.success is False
    assert result.evidence["failure_class"] == "evaluator_failure"
