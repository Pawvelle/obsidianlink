# ObsidianLink Roadmap

> Full research and development plans are available in `docs/plans/`.

## Current Phase

**Phase 1 — Minimal Minecraft Agent Loop**

Goal: establish the first real closed loop between Python, Minecraft, observation, and action.

## Current Task

**Step 1 — Minimal Real Environment Adapter**

Implement the minimum environment integration required to:

* start and reset the Minecraft / MineRL environment;
* obtain one real RGB observation;
* execute one bounded action;
* obtain the next observation;
* close the environment cleanly.

Do not implement Benchmark tasks, LLM agents, planners, evaluators, or Multi-Agent features in this step.

## Completed

* Research direction frozen
* Research-First Master Plan frozen
* Development Plan frozen
* Old v2 implementation removed from the active codebase
* Minimal Research-First project skeleton established
* Minimal Environment / Agent / Task / Evaluator / Runner interfaces established
* Offline smoke-test structure established

## Next

After the real environment adapter works:

1. connect RGB observation to the Agent-visible `Observation`;
2. support the minimum bounded action set required for the first Agent loop;
3. connect one real `ModelClient`;
4. run the first real `Observation -> Agent -> Action -> Minecraft` loop.

## Blocked

None.
