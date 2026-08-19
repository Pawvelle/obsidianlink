# ObsidianLink 本地开发环境

本文件说明 ObsidianLink 应使用的本机开发环境。供 AI Coding Agent 与后续开发者使用。不要把它当成完整开发手册。

## 1. Development Environment

开发、测试和运行统一使用现有 Conda 环境 `mc-agent`，不要使用系统 Python。

交互式：

```bash
conda activate mc-agent
```

非交互 / AI Agent 命令：

```bash
conda run -n mc-agent python ...
```

若当前 shell 找不到 `conda`，使用：

```bash
/opt/anaconda3/bin/conda run -n mc-agent python ...
```

不要使用 `/usr/bin/python3` 或未激活的系统 Python。

## 2. Verified Runtime

以下为本机 2026-08-18 实际检查结果，不是猜测。

| 项 | 已确认值 |
| --- | --- |
| Conda 环境 | `mc-agent`（`/opt/anaconda3/envs/mc-agent`） |
| Python | 3.10.20（conda-forge） |
| Python executable | `/opt/anaconda3/envs/mc-agent/bin/python` |
| Java（`mc-agent` 内，MineRL 应使用这个） | Zulu OpenJDK 1.8.0_472（conda-forge `openjdk` 8.0.472） |
| Java executable | `/opt/anaconda3/envs/mc-agent/bin/java` |
| `JAVA_HOME`（激活 `mc-agent` 后） | `/opt/anaconda3/envs/mc-agent` |
| MineRL | 1.0.2（`import minerl` 来自 `mc-agent` site-packages） |
| Gym | 0.23.1 |
| NumPy | 1.23.5 |
| OpenCV | opencv-python 4.8.1.78 |
| Pillow | 11.3.0 |
| pytest | 9.1.1 |

补充：

- 系统默认 Java 是 Temurin OpenJDK 25.0.3。这不是 MineRL 运行时。MineRL / Minecraft 必须使用 `mc-agent` 内的 Java 8。
- MineRL 1.0.2 对应 Minecraft 1.16.5（来自已安装包说明，以及 2026-07-16 历史 crash report）。本次检查没有重新启动 Minecraft。
- Phase 1 起仓库已接入 MineRL：`obsidianlink.env.minerl.MineRLEnvironment` 与 `ControlledSceneEnv`。`import minerl` 仍来自 `mc-agent` site-packages，不指向 `vendor/minerl`。
- `vendor/minerl` 在本机存在，但被 `.gitignore` 忽略。

## 3. Environment Rules

- 优先复用现有 `mc-agent`。不要擅自创建新的 Conda / venv 环境。
- 不要全局安装 Python 包。
- 安装依赖前先检查 `mc-agent` 是否已有。
- 不要未经必要性验证就升级 Python、Java、MineRL、Gym 或 NumPy。
- 不要为了“现代化”而升级当前可运行的 Minecraft / MineRL 技术栈。`import gym` 可能打印 unmaintained 警告；不要因此改用 Gymnasium 或升级 NumPy 2。
- `vendor/minerl` 优先视为已有运行时依赖，不要主动大规模重构。若必须修改，只做解决实际阻塞问题所需的最小改动。
- 当前有效依据是：当前仓库代码、当前可运行的 `mc-agent`、以及 `docs/plans/` 中的 Research-First 计划。不要把已废弃的旧 v2 架构或 P0–P8 / E0–E12 流程写回本项目。

## 4. Before Running

```bash
conda activate mc-agent
python --version
python -c "import sys; print(sys.executable)"
java -version
python -c "import os; print(os.environ.get('JAVA_HOME'))"
python -c "import minerl, gym, numpy; print('minerl/gym/numpy import ok')"
```

非交互等价命令：

```bash
conda run -n mc-agent python --version
conda run -n mc-agent python -c "import sys; print(sys.executable)"
conda run -n mc-agent java -version
conda run -n mc-agent python -c "import minerl, gym, numpy; print('minerl/gym/numpy import ok')"
```

期望：Python 3.10.20、executable 指向 `.../envs/mc-agent/bin/python`、Java `1.8.0_472`。不要启动 MineRL / Minecraft 环境来做常规环境检查。

## 5. Project Development Principle

- Research First，Benchmark First。
- 先跑通真实实验，再增加框架。
- 当前阶段用不到的基础设施不创建。
- 避免无关的大规模重构。
- 优先选择最短路径完成当前实验目标。

当前阶段：**Phase 2 — Benchmark MVP 已完成。下一阶段是 Phase 3 — Single-Agent Portal Benchmark。**

当前目标：在用户明确要求时实现 **L1 Controlled Construction**。不要再增加 D1 / D2 / D3 diagnostic task。不要提前开发 D4 / D5 / D6。不要把 motor 写回 D2。未接到实现 L1 的明确指令时，不要开始写 L1 代码。
