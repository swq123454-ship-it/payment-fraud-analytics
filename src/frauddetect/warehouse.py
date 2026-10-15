"""SQLite warehouse builder.

Transformations live in ``sql/*.sql`` and run in filename order, so the whole
feature layer can be rebuilt from raw rows in one command and reviewed as a
diff instead of being buried in notebook cells.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config as cfg


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path or cfg.DB_PATH)
    # The window functions in 03_features.sql sort within partitions. Giving
    # SQLite a bigger page cache keeps those sorts in memory.
    connection.execute("PRAGMA temp_store = MEMORY;")
    connection.execute("PRAGMA cache_size = -200000;")
    return connection


def execute_sql_file(connection: sqlite3.Connection, path: Path) -> None:
    """Execute a .sql file statement by statement.

    ``executescript`` is deliberately avoided: it wraps the file in one implicit
    transaction and reports a single opaque failure, which makes a broken
    transformation far harder to locate.
    """
    script = path.read_text(encoding="utf-8")
    statements = [s.strip() for s in script.split(";") if s.strip()]
    for statement in statements:
        cleaned = "\n".join(
            line for line in statement.splitlines() if not line.strip().startswith("--")
        ).strip()
        if not cleaned:
            continue
        try:
            connection.execute(cleaned)
        except sqlite3.Error as error:  # pragma: no cover - surfaced to the caller
            preview = cleaned[:400] + ("..." if len(cleaned) > 400 else "")
            raise RuntimeError(f"{path.name}: {error}\n\n{preview}") from error


def build_warehouse(extract, verbose: bool = True, db_path: Path | None = None) -> Path:
    """Load the raw tables, then run every SQL script in order.

    ``db_path`` is exposed so that tests can build a throwaway warehouse instead
    of overwriting the one the pipeline just produced.
    """
    db_path = Path(db_path or cfg.DB_PATH)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = get_connection(db_path)
    try:
        extract.transactions.to_sql("raw_transactions", connection, if_exists="replace", index=False)
        extract.identity.to_sql("raw_identity", connection, if_exists="replace", index=False)

        sql_files = sorted(cfg.SQL_DIR.glob("*.sql"))
        if not sql_files:
            raise RuntimeError(f"No SQL scripts found in {cfg.SQL_DIR}")
        for path in sql_files:
            execute_sql_file(connection, path)
            if verbose:
                print(f"  [sql] applied {path.name}")
        connection.commit()
    finally:
        connection.close()
    return db_path


def read_table(name: str, db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as connection:
        return pd.read_sql_query(f"SELECT * FROM {name}", connection)


def list_tables(db_path: Path | None = None) -> list[str]:
    query = "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    with get_connection(db_path) as connection:
        return [row[0] for row in connection.execute(query).fetchall()]


def table_counts(db_path: Path | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    with get_connection(db_path) as connection:
        for name in list_tables(db_path):
            if name.startswith("sqlite_"):
                continue
            counts[name] = connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    return counts
