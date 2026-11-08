#!/bin/sh
# 云端部署启动脚本（WorkBuddy sites / 通用 PaaS）
# 平台把公网端口注入 PORT 环境变量；本地调试可用 bash start.sh
exec python3 -m streamlit run app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.fileWatcherType none
