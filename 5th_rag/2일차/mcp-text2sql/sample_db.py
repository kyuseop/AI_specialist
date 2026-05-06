"""
샘플 데이터베이스 생성 스크립트
실행: python sample_db.py
결과: company.db 파일이 생성됩니다
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company.db")

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# ── 테이블 생성 ──
c.execute("""CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    position TEXT NOT NULL,
    salary INTEGER NOT NULL,
    hire_date TEXT NOT NULL
)""")

c.execute("""CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price INTEGER NOT NULL
)""")

c.execute("""CREATE TABLE IF NOT EXISTS sales (
    id INTEGER PRIMARY KEY,
    product_id INTEGER NOT NULL,
    employee_id INTEGER NOT NULL,
    amount INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    sale_date TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id),
    FOREIGN KEY (employee_id) REFERENCES employees(id)
)""")

# ── 직원 데이터 ──
employees = [
    (1, "김철수", "개발팀", "시니어 개발자", 6500, "2022-03-15"),
    (2, "이영희", "마케팅팀", "팀장", 7200, "2021-01-10"),
    (3, "박지민", "개발팀", "주니어 개발자", 4200, "2024-06-01"),
    (4, "최수진", "영업팀", "매니저", 5800, "2022-09-20"),
    (5, "정민호", "개발팀", "테크 리드", 8500, "2020-04-05"),
    (6, "한소영", "마케팅팀", "디자이너", 4800, "2023-11-12"),
    (7, "오준서", "영업팀", "영업 사원", 4000, "2024-01-15"),
    (8, "윤지혜", "인사팀", "HR 매니저", 5500, "2021-07-20"),
    (9, "강태현", "개발팀", "백엔드 개발자", 5800, "2023-02-28"),
    (10, "서미래", "영업팀", "팀장", 7000, "2020-11-01"),
]

# ── 제품 데이터 ──
products = [
    (1, "클라우드 호스팅 Basic", "클라우드", 50000),
    (2, "클라우드 호스팅 Pro", "클라우드", 150000),
    (3, "데이터 분석 플랫폼", "데이터", 300000),
    (4, "AI 챗봇 솔루션", "AI", 500000),
    (5, "보안 모니터링", "보안", 200000),
    (6, "API Gateway", "클라우드", 100000),
    (7, "ML 파이프라인", "AI", 400000),
    (8, "로그 분석기", "데이터", 80000),
]

# ── 매출 데이터 (2025-2026) ──
sales = [
    (1, 1, 4, 50000, 1, "2025-07-15"),
    (2, 2, 4, 300000, 2, "2025-08-20"),
    (3, 4, 10, 1500000, 3, "2025-09-10"),
    (4, 3, 7, 300000, 1, "2025-10-05"),
    (5, 5, 4, 400000, 2, "2025-10-22"),
    (6, 1, 7, 150000, 3, "2025-11-03"),
    (7, 2, 10, 450000, 3, "2025-11-15"),
    (8, 6, 4, 200000, 2, "2025-12-01"),
    (9, 4, 10, 1000000, 2, "2025-12-18"),
    (10, 7, 7, 400000, 1, "2026-01-10"),
    (11, 3, 4, 600000, 2, "2026-01-25"),
    (12, 8, 10, 240000, 3, "2026-02-05"),
    (13, 2, 7, 150000, 1, "2026-02-14"),
    (14, 5, 4, 600000, 3, "2026-02-28"),
    (15, 4, 10, 2000000, 4, "2026-03-05"),
    (16, 1, 7, 100000, 2, "2026-03-12"),
    (17, 6, 4, 300000, 3, "2026-03-18"),
    (18, 7, 10, 800000, 2, "2026-03-19"),
]

# ── 데이터 삽입 ──
c.execute("DELETE FROM sales")
c.execute("DELETE FROM products")
c.execute("DELETE FROM employees")

c.executemany("INSERT INTO employees VALUES (?,?,?,?,?,?)", employees)
c.executemany("INSERT INTO products VALUES (?,?,?,?)", products)
c.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", sales)

conn.commit()
conn.close()

print(f"✅ 데이터베이스 생성 완료: {DB_PATH}")
print(f"   - employees: {len(employees)}명")
print(f"   - products: {len(products)}개")
print(f"   - sales: {len(sales)}건")
