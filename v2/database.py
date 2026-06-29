import os
import sqlite3
from contextlib import contextmanager
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_db.db")

@contextmanager
def get_conn():
    conn=sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    schema="""
    CREATE TABLE IF NOT EXISTS docs(
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    parent_folder TEXT NOT NULL,
    file_extension TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY ,
    doc_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    vector BLOB NOT NULL,
    FOREIGN KEY(doc_id) REFERENCES docs(id) ON DELETE CASCADE
    );
    """
    with get_conn() as temp:
        temp.executescript(schema)
        temp.commit()
if __name__=="__main__":
    init_db()
    print("Database initialised successfully!")