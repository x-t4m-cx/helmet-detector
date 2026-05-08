import sqlite3
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class FaceStatsRepository:
    """SQLite-хранилище статистики обработки лиц."""

    def __init__(self, db_path="face_stats.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    frame_num INTEGER,
                    total_faces INTEGER NOT NULL,
                    with_helmet INTEGER NOT NULL,
                    without_helmet INTEGER NOT NULL,
                    compliance REAL NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS face_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    has_helmet INTEGER NOT NULL,
                    score REAL NOT NULL,
                    x INTEGER NOT NULL,
                    y INTEGER NOT NULL,
                    w INTEGER NOT NULL,
                    h INTEGER NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES processing_events(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_processing_events_created_at ON processing_events(created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_face_detections_event_id ON face_detections(event_id)"
            )
            self._conn.commit()

    def save_event(self, source, total_faces, with_helmet, without_helmet, compliance, results, frame_num=None):
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                INSERT INTO processing_events (
                    created_at, source, frame_num, total_faces,
                    with_helmet, without_helmet, compliance
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, source, frame_num, total_faces, with_helmet, without_helmet, float(compliance)),
            )
            event_id = cursor.lastrowid

            face_rows = []
            for r in results:
                x, y, w, h = r.get("position", (0, 0, 0, 0))
                face_rows.append(
                    (
                        event_id,
                        1 if r.get("has_helmet", False) else 0,
                        float(r.get("score", 0.0)),
                        int(x),
                        int(y),
                        int(w),
                        int(h),
                    )
                )

            if face_rows:
                cursor.executemany(
                    """
                    INSERT INTO face_detections (
                        event_id, has_helmet, score, x, y, w, h
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    face_rows,
                )

            self._conn.commit()
        logger.debug("Сохранена статистика в SQLite: source=%s total=%s frame=%s", source, total_faces, frame_num)

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def get_recent_events(self, limit=50):
        """Возвращает последние события обработки для отображения истории."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT
                    id,
                    created_at,
                    source,
                    frame_num,
                    total_faces,
                    with_helmet,
                    without_helmet,
                    compliance
                FROM processing_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
