# MCP Text2SQL Server

자연어로 SQLite 데이터베이스를 쿼리할 수 있는 MCP 서버입니다.

## 빠른 시작

### 1. 의존성 설치

```bash
pip install mcp
```

### 2. 샘플 데이터베이스 생성

```bash
python sample_db.py
```

### 3. Claude Desktop 설정

`claude_desktop_config.json` 파일에 아래 내용을 추가하세요:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "text2sql": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/mcp-text2sql"
    }
  }
}
```

> `cwd` 경로를 이 폴더의 실제 경로로 변경하세요.

### 4. Claude Desktop 재시작 후 사용

## 사용 가능한 도구

| 도구 | 설명 |
|------|------|
| `get_schema` | DB 테이블 구조 조회 |
| `query_db` | SQL SELECT 쿼리 실행 (읽기 전용) |
| `list_tables` | 테이블 목록 조회 |
| `sample_data` | 테이블 샘플 데이터 미리보기 |

## 예시 질문

- "우리 데이터베이스에 어떤 테이블이 있어?"
- "개발팀 직원 목록을 보여줘"
- "부서별 평균 급여를 알려줘"
- "매출이 가장 높은 직원 3명은?"
- "AI 카테고리 제품의 총 매출은?"
