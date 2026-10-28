import sqlite3
from contextlib import contextmanager
from backend.config import DATABASE_PATH

_initialized = False

def init_db() -> None:
    global _initialized
    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                tool_names TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    _initialized = True

@contextmanager
def connection():
    if not _initialized:
        init_db()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def save_task(prompt: str, response: str, tool_names: list[str]) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO task_history(prompt, response, tool_names) VALUES (?, ?, ?)",
            (prompt, response, ", ".join(tool_names)),
        )

def get_recent_tasks(limit: int = 50) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, prompt, response, tool_names, created_at FROM task_history ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
    return [dict(row) for row in rows]

def add_note(title: str, content: str) -> int:
    with connection() as conn:
        cur = conn.execute("INSERT INTO notes(title, content) VALUES (?, ?)", (title, content))
        return int(cur.lastrowid)

def get_notes(limit: int = 100) -> list[dict]:
    with connection() as conn:
        rows = conn.execute("SELECT id, title, content, created_at FROM notes ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
