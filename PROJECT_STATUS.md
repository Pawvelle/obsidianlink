# 当前状态

更新时间：2026-08-05

## 当前唯一目标

任务：`R2-CAPABILITY-MANIFEST`

为 `casting_c1_fixed` 增加一个纯离线的后端能力清单，用类型和测试回答：

- 是否支持选择并使用水桶、熔岩桶；
- 是否能观察公开库存和当前手持物品；
- evaluator 是否能读取目标方块和流体真值；
- 缺少关键能力时能否在 episode 开始前停止。

## 建议修改位置

- 新增 `obsidianlink/env/capabilities.py`
- 新增 `tests/test_capabilities.py`
- 如有必要，只对 FakeBackend 增加最小接口

不要实现 casting evaluator、driver 或 VLM。

## 交付内容

1. 一个不可变、类型明确的 `BackendCapabilities` 数据对象。
2. 必要能力缺失时返回明确的缺失项。
3. FakeBackend 支持与不支持两组测试。
4. 全部离线测试通过。
5. 完成后把下一任务改为 `R3-CASTING-EVALUATOR`。

## 已完成

- 核心类型、动作白名单、FakeBackend 和结构化日志；
- MineRL 单角色生命周期与 Portal 环境桥接；
- Portal 框架、激活和进入下界的自动评估；
- 活动任务 `benchmark/instances/active/casting_c1_fixed.json`；
- 离线实验契约 `configs/experiments/active/casting_c1_contract.json`。

## 当前限制

- `casting_c1_fixed` 还没有 evaluator 或 driver；
- 后端能力尚未确认；
- 当前禁止真实 MineRL、Gradle 和模型调用。

## 下一 Agent 直接执行

> 实现 `BackendCapabilities` 和对应 FakeBackend 离线测试。完成后运行全套测试并更新本文件。
