# ObsidianLink v2.0 Nether Portal Construction Dataset Card（计划）

当前不存在正式 v2 数据集。本文件冻结未来 episode/evidence 的语义；它不能被引用为已有 benchmark data。

## Unit of data

一个数据单元是一条冻结 task instance 下的完整 episode。正式记录必须能区分：

- benchmark/task/generator/evaluator/code version；
- family=`nether_portal_construction`、suite、mode、level、layout、split；
- world seed 与公开 variation profile；
- task 指定 Agent 与预算；
- observation/action/message/evaluation/log 的 `episode_id`、`step_id`、适用时的 `agent_id`；
- portal construction、activation、Nether entry 与 episode attribution；
- success、milestones、Completion Rate、效率、恢复与 evidence completeness；
- verification level。

## Evidence layout

正式 episode 至少保存：

```text
task_instance.json
experiment_config.json
capability_manifest.json
code_version.json
evaluator_version.json
initial.png
final.png
events.jsonl
evaluator_events.jsonl
summary.json
manual_review.md
```

Agent-visible events 与 evaluator-only events 必须物理或逻辑隔离。缺 server truth、identity、step ordering、portal attribution 或版本时 fail closed。

## Planned fields

核心字段包括 `benchmark_version`、`task_instance_id`、`task_family`、`suite`、`agent_mode`、`task_level`、`layout_type`、`split`、`world_seed`、`variation_profile`、`agent_ids`、`designated_agent_ids`、`success`、`completion_rate`、`environment_steps`、`game_time_seconds`、`model_calls`、`failure_type`、`evaluator_version`、`verification_level`。

`task_level` 的 v2 命名空间为 Diagnostic D1–D6 或 End-to-End L1–L4；Roadmap P0–P8 与 Environment Validation E0–E12 不是 dataset task levels。当前 `obsidianlink.core.types.TaskInstance` 及其 route/workflow 字段属于 v1 compatibility，不是这里规划的 canonical v2 episode contract。

Multi-Agent 追加显式 sender/recipient、accepted message、role condition、communication counts、makespan、idle/duplicate work 和 per-agent contribution；不能保存隐式共享的私有 observation/inventory/memory。

## Splits and leakage

未来使用 `train` / `dev` / `test`。同源模板、近重复世界和 seed 不跨 split。Test 隐藏参数和 evaluator truth 不进入 Agent input。当前没有冻结 generator 或 split assignment。

## Verification semantics

- `unit_verified` 数据只能来自 parser/schema/pure evaluator/FakeBackend/regression，不得混入真实 capability 或 leaderboard；
- `integration_verified` 必须来自真实 MineRL/Minecraft；
- `benchmark_evaluated` 必须来自冻结 benchmark experiments。

FakeBackend success 不得转换或标注为真实 Minecraft success。历史 v1 runs 与 task instances 只作 legacy/calibration/regression，不能混入 v2 Success Rate。

## Not stored

不保存 API key、访问令牌、模型权重、隐藏推理、与 episode 无关的个人数据、未经协议允许的 evaluator truth 副本或跨 Agent 私有数据。

## Current status

只有 v2 schema/interface 计划和 legacy evidence。P1 E0–E12 尚未在本次重构中真实运行；L1–L4 尚未实现，没有 E10 `integration_verified` evidence，没有 end-to-end benchmark episodes，没有 Multi-Agent gameplay data，也没有 `benchmark_evaluated` dataset。
