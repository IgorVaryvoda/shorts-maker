import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MusicDatabase:
    def __init__(self, db_path: str = ".cache/music_library.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS music_usage (
                    id INTEGER PRIMARY KEY,
                    filepath TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    last_used REAL NOT NULL,
                    usage_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_music_usage_filepath
                ON music_usage (filepath);
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_music_usage_last_used
                ON music_usage (last_used);
            """)
            # Tables for JSON-based caches
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    filepath TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS library_analysis (
                    filepath TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)

    def track_music_usage(self, filepath: str):
        filename = os.path.basename(filepath)
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO music_usage (
                    filepath, filename, last_used, usage_count
                )
                VALUES (?, ?, ?, 1)
                ON CONFLICT(filepath) DO UPDATE SET
                    last_used = excluded.last_used,
                    usage_count = usage_count + 1
                """,
                (filepath, filename, time.time())
            )

    def get_usage_count(self, filepath: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT usage_count FROM music_usage WHERE filepath = ?",
            (filepath,)
        )
        result = cursor.fetchone()
        return result['usage_count'] if result else 0

    def get_recently_used(self, limit: int) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT filename FROM music_usage
            ORDER BY last_used DESC
            LIMIT ?
            """,
            (limit,)
        )
        return [row['filename'] for row in cursor.fetchall()]

    def get_all_usage_stats(self) -> List[sqlite3.Row]:
        """
        Returns all usage stats, useful for migration or debugging.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM music_usage ORDER BY usage_count DESC")
        return cursor.fetchall()

    def close(self):
        self.conn.close()

    def __del__(self):
        self.close()

    def migrate_json_to_table(self, json_path: str, table_name: str):
        """Migrate a JSON cache file into the specified SQLite table."""
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path) as f:
                cache_data = json.load(f)
            with self.conn:
                for filepath, entry in cache_data.items():
                    self.conn.execute(
                        f"INSERT OR REPLACE INTO {table_name} (filepath, data, last_updated)"
                        " VALUES (?, ?, ?)",
                        (filepath, json.dumps(entry), time.time())
                    )
            os.remove(json_path)
            logger.info(f"Migrated {json_path} to {table_name} table")
        except Exception as e:
            logger.error(f"Failed to migrate {json_path}: {e}")

    def get_analysis_cache(self) -> Dict[str, Any]:
        """Retrieve all analysis_cache entries as a dict."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT filepath, data FROM analysis_cache")
        return {row['filepath']: json.loads(row['data']) for row in cursor.fetchall()}

    def get_library_analysis(self) -> Dict[str, Any]:
        """Retrieve all library_analysis entries as a dict."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT filepath, data FROM library_analysis")
        return {row['filepath']: json.loads(row['data']) for row in cursor.fetchall()}

    def set_analysis_cache(self, cache: Dict[str, Any]):
        """Write analysis_cache dict into the database."""
        with self.conn:
            for filepath, entry in cache.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO analysis_cache (filepath, data, last_updated)"
                    " VALUES (?, ?, ?)",
                    (filepath, json.dumps(entry), time.time())
                )

    def set_library_analysis(self, cache: Dict[str, Any]):
        """Write library_analysis dict into the database."""
        with self.conn:
            for filepath, entry in cache.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO library_analysis (filepath, data, last_updated)"
                    " VALUES (?, ?, ?)",
                    (filepath, json.dumps(entry), time.time())
                )
