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

当前任务仍是 `MineRLBasaltFindCave-v0`。

已完成的个人项目里程碑：

- 安全前进提示与验收门已经落地；零角度 `look/turn` 不再算有效模型动作。
- 2026-07-17 的 5×800 ticks 验收全部通过：25 个有效 Qwen 前进决策、290 个
  forward ticks、每回合至少两种有效动作，`ESC=0`、stale decision=0。
- JSON 动作增加 fail-closed 的 `cave_visible` 判断；只有同时带有暗色、岩石、
  开口和方向证据的声明才记录为洞穴候选帧。
- 历史土墙误报负样本和一个 800-tick 集成回合已经通过；普通平原画面没有产生
  洞穴候选，也没有破坏安全前进闭环。

Phase 4 尚未完成：现有 506 张历史决策帧和最新回放中没有可信的洞穴入口正样本，
因此不能验证正例，也不能把 FindCave 标成成功。下一步只需要获得真实洞穴入口帧或
实际回合，复核候选判断与接近动作；不增加论文式 A/B runner、复杂长期记忆或新的
模型后端。

本地证据位于：

- `runs/phase4-forward-acceptance/20260717-205706/summary.json`
- `runs/phase4-cave-preflight/20260717-211041/summary.json`

## 长期安全边界

1. 模型输出必须经过 JSON 解析、动作白名单和数值限幅。
2. MineRL 环境只能由运行 step loop 的线程操作。
3. Qwen worker 不得阻塞正在运行的 MineRL step loop。
4. 模型权重、MineRL、Python 和 JDK 版本保持锁定。
5. `vendor/minerl` 是独立 Git 仓库，外层仓库不提交或改写其历史。
6. MineRL Gradle 构建会执行第三方代码，重新构建前必须获得用户明确批准。

详细的历史验证记录保存在 `runs/history/EXECUTION_LOG.md`，旧运行资产保存在
`runs/history/artifacts/` 和 `runs/history/logs/`。
