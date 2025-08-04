import sqlite3

conn = sqlite3.connect("data.sqlite3")
cursor = conn.cursor()

# 테이블 삭제
# sql = "drop table worker_alerts"
# cursor.execute(sql)
# conn.commit()

cursor.execute("""
 CREATE TABLE IF NOT EXISTS workers (
   worker_id TEXT PRIMARY KEY,
   name TEXT NOT NULL,
   team_id TEXT NOT NULL,
   position TEXT
 )
 """)

while(True):
    data = input("사용자 정보 입력: ")
    if not data.strip():  # 엔터 누르면 종료
        break

    try:
        words =  data.split(",")
        if len(words) != 4:
            print("정확히 4개의 항목(worker_id, name, team_id, position)을 쉼표로 구분해 입력하세요.")
            continue

        result = '", "'.join(words)
        cursor.execute(
            f'INSERT INTO workers VALUES ("{result}")'
        )
        conn.commit()
        print(f"추가 완료")
    except ValueError:
        print("입력 형식 오류! 정확히 4개의 값을 입력하세요.")
    except sqlite3.IntegrityError:
        print("중복된 worker_id 또는 제약 조건 위반!")
    except Exception as e:
        print("예기치 못한 오류:", e)

conn.close()
print("\n데이터 입력이 종료되었습니다.")