#!/bin/bash
# 下载 models 仓库数据到 src-tauri/resources/models
# 用于本地开发环境

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/src-tauri/resources/models"

BASE_URL="https://raw.githubusercontent.com/aiclientproxy/models/main"

echo "📦 下载 models 数据..."
echo "   目标目录: $MODELS_DIR"

# 创建目录结构
mkdir -p "$MODELS_DIR/providers" "$MODELS_DIR/aliases"

# 下载 index.json
echo "   下载 index.json..."
curl -sL "$BASE_URL/index.json" -o "$MODELS_DIR/index.json"

# 解析 providers 列表并下载每个 provider 的数据
echo "   下载 providers..."
for provider in $(cat "$MODELS_DIR/index.json" | jq -r '.providers[]'); do
  echo "     - $provider"
  curl -sL "$BASE_URL/providers/${provider}.json" -o "$MODELS_DIR/providers/${provider}.json"
done

# 下载别名配置
echo "   下载 aliases..."
for alias in kiro antigravity; do
  echo "     - $alias"
  curl -sL "$BASE_URL/aliases/${alias}.json" -o "$MODELS_DIR/aliases/${alias}.json" 2>/dev/null || echo "     (跳过 $alias - 文件不存在)"
done

# 统计
provider_count=$(ls -1 "$MODELS_DIR/providers" 2>/dev/null | wc -l | tr -d ' ')
alias_count=$(ls -1 "$MODELS_DIR/aliases" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "✅ 下载完成!"
echo "   Providers: $provider_count"
echo "   Aliases: $alias_count"
