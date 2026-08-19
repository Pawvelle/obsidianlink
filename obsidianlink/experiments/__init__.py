"""Experiment scripts that require heavy resources (real models,
live MineRL, GPU) live here. They are NOT part of the regular
unit-test suite; each one is a standalone script with its own
``argparse`` entry point and ``main()``.

See:

* :mod:`obsidianlink.experiments.smoke_qwen_vl_d1` — Phase 2B
  synthetic-frame smoke for Qwen3-VL on D1.
* :mod:`obsidianlink.experiments.debug_d2_01_scenes` — D2-01
  spawn-yaw frame capture (no VLM, no motor).
* :mod:`obsidianlink.experiments.run_d2_01` — D2-01 Direction
  Grounding live left / center / right evaluation.
* :mod:`obsidianlink.experiments.debug_d2_02_scenes` — D2-02
  3×3 spawn-pose frame capture (no VLM, no motor).
* :mod:`obsidianlink.experiments.run_d2_02` — D2-02 Spatial
  Region Grounding live 3×3 evaluation.
* :mod:`obsidianlink.experiments.debug_d3_01_scenes` — D3-01
  spawn-yaw capture + camera-sign / hidden-yaw check.
* :mod:`obsidianlink.experiments.run_d3_01` — D3-01 Camera
  Alignment live left / center / right evaluation.
* :mod:`obsidianlink.experiments.debug_d3_02_scenes` — D3-02
  start-frame capture + scripted forward / hidden-distance check.
* :mod:`obsidianlink.experiments.run_d3_02` — D3-02 Target
  Approach live evaluation.
"""
