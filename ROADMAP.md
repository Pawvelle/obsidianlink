# mc-agent Roadmap

`mc-agent` 是一个在本机运行的个人 Minecraft Agent 项目：MineRL 提供
Minecraft 1.16.5 环境，Qwen3-VL-2B-Instruct 读取第一人称画面并输出受约束的
JSON 宏动作，本地执行器负责安全地转换成键盘、鼠标和相机操作。

## 固定技术栈

- Python 3.10.20，Conda 环境 `mc-agent`
- OpenJDK 8.0.472
- MineRL 1.0.2 / Minecraft 1.16.5 / Gym 0.23.1 / NumPy 1.23.5
- Qwen3-VL-2B-Instruct，锁定本地权重，Apple MPS / FP16
- MineRL 源码位于独立嵌套仓库 `vendor/minerl`

这些版本不会在普通功能开发或结构整理中顺手升级。

## Phase 1 — 运行 Minecraft + MineRL

**状态：已完成。**

- Apple Silicon MineRL 构建和补丁已完成。
- `MineRLBasaltFindCave-v0` 可以 reset、step、close。
- Minecraft 能自动关闭且不会留下运行进程。

## Phase 2 — Qwen 理解 Minecraft 画面

**状态：已完成。**

- Qwen 从本地锁定模型加载。
- 在 Apple MPS 上完成 Minecraft 首帧视觉推理。
- 输出经过严格 JSON 动作解析，不执行模型生成的代码或命令。

## Phase 3 — Agent 控制角色行动

**状态：已完成。**

- MineRL step loop 与 Qwen 推理解耦，模型慢速推理不会阻塞环境循环。
- 已实现动作白名单、数值限幅、watchdog、episode barrier、decision-ack、
  简单 memory 和结构化日志。
- 已完成多个可回放闭环 episode，`ESC` 默认保持关闭。

## Phase 4 — 完成简单任务

**状态：进行中。**

当前任务仍是 `MineRLBasaltFindCave-v0`。闭环运行已经稳定，但 Agent 还没有可靠地
完成寻找洞穴。下一步应优先改善可观察的持续移动和简单目标判断，而不是继续增加
论文式 A/B runner、复杂长期记忆或新的模型后端。

## 长期安全边界

1. 模型输出必须经过 JSON 解析、动作白名单和数值限幅。
2. MineRL 环境只能由运行 step loop 的线程操作。
3. Qwen worker 不得阻塞正在运行的 MineRL step loop。
4. 模型权重、MineRL、Python 和 JDK 版本保持锁定。
5. `vendor/minerl` 是独立 Git 仓库，外层仓库不提交或改写其历史。
6. MineRL Gradle 构建会执行第三方代码，重新构建前必须获得用户明确批准。

详细的历史验证记录保存在 `runs/history/EXECUTION_LOG.md`，旧运行资产保存在
`runs/history/artifacts/` 和 `runs/history/logs/`。
