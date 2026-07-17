# mc-agent

在 Apple Silicon Mac 上，以 MineRL 1.0.2 提供 Minecraft 1.16.5 环境，以
Qwen3-VL-2B-Instruct 读取第一人称画面并输出受约束的游戏动作。

- 唯一执行基线：[MASTER_PLAN.md](MASTER_PLAN.md)
- 当前进度：[docs/EXECUTION_LOG.md](docs/EXECUTION_LOG.md)
- Conda 环境：[environment.yml](environment.yml)
- 模型锁定信息：[config/model.lock.json](config/model.lock.json)

快速检查：

```bash
conda activate mc-agent
python scripts/check_environment.py
```

MineRL 的第三方 Gradle 构建已在获得明确批准后完成，并通过主规划 Phase 1
验收；后续运行复用已验证产物，不重复构建。

## 仓库边界

- 外层 `mc-agent` 保存智能体代码、配置、测试和实验记录，是独立的本地 Git 仓库；当前尚未创建初始提交或配置 GitHub remote。
- `vendor/minerl` 是锁定在上游 `minerllabs/minerl` 的独立嵌套 Git checkout；外层通过 `.gitignore` 排除它，避免误提交其构建产物或本地 Apple Silicon 补丁。
- MineRL 的固定分支、提交与本地补丁证据以 `MASTER_PLAN.md` 和 `docs/EXECUTION_LOG.md` 为准。
