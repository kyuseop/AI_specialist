"""
MCP Text2SQL Server
SQLite 데이터베이스에 자연어로 질의할 수 있는 MCP 서버

사용법:
  1. pip install mcp
  2. python sample_db.py  (최초 1회)
  3. Claude Desktop config에 등록 후 사용
"""
from mcp.server.fastmcp import FastMCP
import sqlite3
import os

# ── MCP 서버 인스턴스 생성 ──
mcp = FastMCP("text2sql")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company.db")


def get_connection():
    """SQLite DB 연결을 반환합니다"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 결과를 딕셔너리처럼 접근 가능
    return conn


@mcp.tool()
def get_schema() -> str:
    """데이터베이스의 전체 테이블 구조(스키마)를 반환합니다.
    어떤 테이블과 컬럼이 있는지 파악할 때 사용하세요."""
    conn = get_connection()
    tables = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    conn.close()
    return "\n\n".join(row["sql"] for row in tables)


@mcp.tool()
def query_db(sql: str) -> str:
    """SQL 쿼리를 실행하고 결과를 반환합니다.
    SELECT 쿼리만 허용됩니다 (읽기 전용).

    Args:
        sql: 실행할 SQL SELECT 쿼리
    """
    # 보안: SELECT만 허용
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "Error: SELECT 쿼리만 허용됩니다. 데이터 수정은 불가합니다."

    # 위험 키워드 차단
    dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "EXEC"]
    for keyword in dangerous:
        if keyword in sql_upper:
            return f"Error: '{keyword}' 키워드가 포함된 쿼리는 실행할 수 없습니다."

    conn = get_connection()
    try:
        rows = conn.execute(sql).fetchall()
        if not rows:
            return "결과가 없습니다."
        result = [dict(r) for r in rows]
        return str(result)
    except Exception as e:
        return f"SQL Error: {e}"
    finally:
        conn.close()


@mcp.tool()
def list_tables() -> str:
    """데이터베이스에 있는 모든 테이블 이름을 반환합니다."""
    conn = get_connection()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    conn.close()
    return ", ".join(row["name"] for row in tables)


@mcp.tool()
def sample_data(table_name: str, limit: int = 5) -> str:
    """특정 테이블의 샘플 데이터를 반환합니다.
    테이블 내용을 미리 확인할 때 유용합니다.

    Args:
        table_name: 조회할 테이블 이름
        limit: 반환할 행 수 (기본값: 5)
    """
    # 테이블명 검증 (SQL Injection 방지)
    conn = get_connection()
    valid_tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]

    if table_name not in valid_tables:
        conn.close()
        return f"Error: '{table_name}' 테이블이 존재하지 않습니다. 사용 가능: {', '.join(valid_tables)}"

    rows = conn.execute(f"SELECT * FROM {table_name} LIMIT ?", (min(limit, 20),)).fetchall()
    conn.close()

    if not rows:
        return f"'{table_name}' 테이블에 데이터가 없습니다."
    return str([dict(r) for r in rows])


# ── 서버 실행 ──
if __name__ == "__main__":
    mcp.run(transport="stdio")
