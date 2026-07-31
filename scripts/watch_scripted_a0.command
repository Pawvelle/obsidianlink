#!/bin/zsh

set -u

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
PYTHON_BIN="/opt/anaconda3/envs/mc-agent/bin/python"

cd "$PROJECT_DIR" || exit 1

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "未找到 MineRL Python: $PYTHON_BIN"
  echo "请确认 mc-agent Conda 环境仍位于 /opt/anaconda3/envs/mc-agent。"
  read -r "?按回车关闭窗口..."
  exit 1
fi

export JAVA_HOME="/opt/anaconda3/envs/mc-agent"
export PATH="/opt/anaconda3/envs/mc-agent/bin:/usr/bin:/bin:/usr/sbin:/sbin"

echo "正在启动可视 MineRL 测试。Minecraft 窗口出现后请不要点击或按键。"
echo "运行结果将写入: $PROJECT_DIR/runs/phase3-scripted-a0-manual-watch"
echo

"$PYTHON_BIN" scripts/run_scripted_a0.py \
  --watch \
  --output-root runs/phase3-scripted-a0-manual-watch
status=$?

echo
if [[ $status -eq 0 ]]; then
  echo "测试完成。"
else
  echo "测试未成功完成，退出码: $status"
fi
echo "结果目录: $PROJECT_DIR/runs/phase3-scripted-a0-manual-watch"
read -r "?按回车关闭窗口..."
exit $status
