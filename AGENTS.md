# ObsidianLink 本地开发环境

本文件规定 ObsidianLink 的当前开发运行时。它服务于 MineDojo 主线，不是完整开发手册。

## 1. 环境

所有开发、测试与运行使用现有 Conda 环境 `mc-agent`，禁止系统 Python 或新建 Conda/venv：

```bash
conda activate mc-agent
# 或
/opt/anaconda3/bin/conda run -n mc-agent python ...
```

先检查已有包；不要全局安装，也不要为“现代化”升级 Python、Gym、NumPy 或 Java。

## 2. 当前 MineDojo 运行时

| 项 | 当前约束 |
| --- | --- |
| Python | `mc-agent` 的 Python 3.10 |
| 环境平台 | MineDojo 0.1 |
| Gym / NumPy | 沿用 `mc-agent` 中已验证的版本，不强制降级或升级 |
| Apple Silicon Java | MineDojo runtime 自动选择 Rosetta x86_64 Java 8，以加载 Minecraft 1.11.2 的 LWJGL 原生库 |
| MineRL | 已卸载；仓库中的相关代码仅作历史归档 |

MineDojo 的 MixinGradle 兼容修复与 Java 选择由 `obsidianlink.env.minedojo_runtime` 管理。不要手工修改 site-packages、替换其校验资源或绕开该运行时准备步骤。

最小检查：

```bash
/opt/anaconda3/bin/conda run -n mc-agent python --version
/opt/anaconda3/bin/conda run -n mc-agent python -c "import minedojo; print('minedojo import ok')"
PYTHONPATH=. /opt/anaconda3/bin/conda run -n mc-agent python -m pytest tests/test_minedojo.py
```

不要仅为常规检查启动 Minecraft。真实 smoke 或可视化试验使用 `obsidianlink.experiments.run_minedojo_smoke` 与 `run_minedojo_harvest_log`。

## 3. 平台与代码规则

- 新环境、runner、测试和 Agent 能力必须使用 `MineDojoEnvironment`。
- 保留 `obsidianlink/env/minerl.py`、Portal benchmark 与历史 runner，但不为它们加功能、不把它们重新接入默认 API。
- 默认 Agent skill surface 保持 primitive-only；复杂任务由 Planner 组合原子动作。
- Agent 只能读取 `Observation` 的 agent-visible 字段；reward、done、任务内部 metadata 和 evaluator truth 不可泄露给 Agent。
- 先跑通最小真实实验，再增加框架；当前实验不需要的基础设施不创建。
- 原始视频帧、trace、运行日志不提交 Git；提交脚本、测试、小型 summary 与文档即可。

## 4. 文档状态

当前不维护仓库内路线图或长期计划文档。MineRL 历史边界见
`docs/LEGACY_MINERL_ARCHIVE.md`。
