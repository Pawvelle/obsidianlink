# MineRL 历史归档

2026-08-22 起，ObsidianLink 的主动开发平台为 MineDojo。仓库仍保留以下 MineRL 相关内容，方便定位历史实验和阅读设计演进：

- `obsidianlink/env/minerl.py`、`scene.py`、`l1_scene.py` 与旧 survival 环境；
- `obsidianlink/benchmark/`、Portal task/evaluator 和对应 runner；
- `tests/test_minerl.py`、`test_l1_*` 及其历史辅助测试；
- `obsidianlink/experiments/HISTORY.md` 中的旧实验记录。

这些内容是**只读归档**：不新增能力、不修复 MineRL runtime、不将其作为 CI 或研究结论的依据。MineRL 已不安装在 `mc-agent` 中，因此直接运行相关脚本或测试会失败；这不构成 MineDojo 主线回归失败。地狱门 Benchmark 的研究目标仍然有效，只是会在 MineDojo 上重新实现。

新的环境、runner、测试、说明和设计决策必须面向 `obsidianlink.env.MineDojoEnvironment`。如果将来确有复现实验的需要，应在隔离环境中恢复旧依赖，而不是污染当前 MineDojo 开发环境。
