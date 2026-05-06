#!/bin/bash
# Claude Desktop config에 text2sql MCP 서버 추가

CONFIG_FILE="$HOME/Library/Application Support/Claude/claude_desktop_config.json"

cat > "$CONFIG_FILE" << 'EOF'
{
  "preferences": {
    "coworkWebSearchEnabled": true,
    "coworkScheduledTasksEnabled": true,
    "ccdScheduledTasksEnabled": true,
    "sidebarMode": "task"
  },
  "mcpServers": {
    "text2sql": {
      "command": "python3",
      "args": ["/Users/wshan/Downloads/mcp-text2sql/server.py"],
      "cwd": "/Users/wshan/Downloads/mcp-text2sql"
    }
  }
}
EOF

echo "✅ 설정 완료! Claude Desktop을 재시작하세요."
cat "$CONFIG_FILE"
