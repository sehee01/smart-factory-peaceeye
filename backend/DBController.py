import sqlite3
from datetime import datetime

conn = sqlite3.connect("data.sqlite3")
cursor = conn.cursor()

# 테이블 삭제
# sql = "drop table ongoing_tasks"
# cursor.execute(sql)
# conn.commit()

# cursor.execute("""
#  CREATE TABLE IF NOT EXISTS workers (
#    worker_id TEXT PRIMARY KEY,
#    name TEXT NOT NULL,
#    team_id TEXT NOT NULL,
#    position TEXT
#  )
#  """)

# while(True):
#     data = input("사용자 정보 입력: ")
#     if not data.strip():  # 엔터 누르면 종료
#         break

#     try:
#         words =  data.split(",")
#         if len(words) != 4:
#             print("정확히 4개의 항목(worker_id, name, team_id, position)을 쉼표로 구분해 입력하세요.")
#             continue

#         result = '", "'.join(words)
#         cursor.execute(
#             f'INSERT INTO workers VALUES ("{result}")'
#         )
#         conn.commit()
#         print(f"추가 완료")
#     except ValueError:
#         print("입력 형식 오류! 정확히 4개의 값을 입력하세요.")
#     except sqlite3.IntegrityError:
#         print("중복된 worker_id 또는 제약 조건 위반!")
#     except Exception as e:
#         print("예기치 못한 오류:", e)

# # 테이블 생성 (없으면)
cursor.execute("""
CREATE TABLE IF NOT EXISTS ongoing_tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_name TEXT NOT NULL,
  part TEXT NOT NULL,
  due_date TEXT NOT NULL,   -- YYYY-MM-DD 또는 ISO 문자열
  details TEXT,
  progress INTEGER DEFAULT 0,
  is_delayed INTEGER DEFAULT 0,   -- 0=false, 1=true
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
)
""")
conn.commit()

print("ongoing_tasks 더미 데이터 입력기")
print('입력 형식: task_name, part, due_date, details')
print('예시: 조립 라인 A, A조, 2025-08-15, 전기 부품 조립 작업')
print('아무것도 입력하지 않고 Enter를 누르면 종료됩니다.\n')

def validate_date(s: str) -> bool:
    s = s.strip()
    # 간단 검증: YYYY-MM-DD 또는 ISO도 일단 허용
    try:
        if "T" in s:
            datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            datetime.strptime(s, "%Y-%m-%d")
        return True
    except Exception:
        return False

while True:
    data = input("업무 입력: ").strip()
    if not data:
        break

    try:
        # 쉼표(,) 기준으로 4개 항목
        parts = [w.strip() for w in data.split(",")]
        if len(parts) != 4:
            print("정확히 4개의 항목(task_name, part, due_date, details)을 쉼표로 구분해 입력하세요.")
            continue

        task_name, part, due_date, details = parts

        if not validate_date(due_date):
            print("due_date 형식이 잘못되었습니다. 예: 2025-08-15 또는 2025-08-15T09:00:00Z")
            continue

        # progress=0, is_delayed=0 기본값으로 삽입
        cursor.execute(
            """
            INSERT INTO ongoing_tasks (task_name, part, due_date, details, progress, is_delayed)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task_name, part, due_date, details, 0, 0)
        )
        conn.commit()
        print("추가 완료")

    except sqlite3.IntegrityError as e:
        print("제약 조건 위반:", e)
    except Exception as e:
        print("예기치 못한 오류:", e)

conn.close()
print("\n데이터 입력이 종료되었습니다.")