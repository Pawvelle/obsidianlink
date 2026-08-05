# 单块黑曜石任务

任务：`casting_c1_fixed`

目标是在固定场景里用水和熔岩生成一块黑曜石。

完整冻结合同见 [`casting_c1_fixed` 任务页](../tasks/casting/casting_c1_fixed.md)。当前分类为 Casting-S-C1 / fixed；`casting_c1_fixed` 是保留的历史兼容 ID。

## 已完成：能力清单

用离线类型声明后端是否支持：

- 选择并使用水桶、熔岩桶；
- 读取 Agent 可见的库存和手持物品；
- evaluator 读取目标 cell 和流体真值；
- 确定性 reset 和有限因果时间窗口。

关键能力缺失时，episode 应在开始前返回明确原因。

## 已完成：evaluator

记录目标 cell 的 before 和 after。合法动作后形成黑曜石才成功；无变化、形成圆石、真值缺失、外部变化或超预算都失败。

## 已完成：确定性 driver

Driver 只能用 Agent 可见信息做决定，并且所有移动、use、等待、重试和总 step 都有上限。先在 FakeBackend 上通过，再申请真实运行。

以上 R2–R4 只在 FakeBackend 离线验证。R5 已在同一隔离原则下完成三个有序 cell 的连续浇筑，详见 [`casting_c3_fixed`](../tasks/casting/casting_c3_fixed.md)。下一工程任务是 `R6-COMPLETE-PORTAL-FRAME`；本 runbook 不代表真实 MineRL 已通过。

## 真实运行前

向用户报告：离线测试结果、缺少的后端能力、是否需要 Gradle/Java 修改、预计运行次数、最长时间、证据目录和清理方式。Gradle 和 MineRL 分别需要明确授权。

## 正式证据目录

```text
runs/casting-c1-fixed/<timestamp>/
  task_instance.json
  experiment_config.json
  capability_manifest.json
  code_version.json
  initial.png
  final.png
  events.jsonl
  evaluator_events.jsonl
  summary.json
  manual_review.md
```
