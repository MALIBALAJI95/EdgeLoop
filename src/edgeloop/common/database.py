"""SQLite database manager for EdgeLoop."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from edgeloop.common.config import get_config


class Database:
    """Manages SQLite persistence for EdgeLoop inferences, drift records, reviews, and models."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            self.db_path = get_config().database_path
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create all required tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Inferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entropy REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    condition TEXT DEFAULT 'unknown',
                    uploaded INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)

            # Drift records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inference_id INTEGER,
                    brightness REAL NOT NULL,
                    contrast REAL NOT NULL,
                    edge_strength REAL NOT NULL,
                    high_freq_std REAL NOT NULL,
                    high_freq_mean_abs REAL NOT NULL,
                    is_drift INTEGER DEFAULT 0,
                    triggered_features TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (inference_id) REFERENCES inferences (id) ON DELETE CASCADE
                )
            """)

            # Selection queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS selection_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inference_id INTEGER NOT NULL,
                    sample_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    upload_reason TEXT NOT NULL,
                    entropy REAL NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (inference_id) REFERENCES inferences (id) ON DELETE CASCADE
                )
            """)

            # Human review records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inference_id INTEGER,
                    sample_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    true_label TEXT NOT NULL,
                    is_corrected INTEGER NOT NULL,
                    reviewer TEXT NOT NULL,
                    notes TEXT,
                    dataset_version TEXT,
                    reviewed_at TEXT NOT NULL,
                    FOREIGN KEY (inference_id) REFERENCES inferences (id) ON DELETE SET NULL
                )
            """)

            # Model registry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    model_name TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    accuracy REAL,
                    macro_f1 REAL,
                    trash_f1 REAL,
                    metrics_json TEXT,
                    approved_by TEXT,
                    promotion_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Dataset versions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    record_count INTEGER NOT NULL,
                    manifest_path TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Indices for rapid querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inf_created ON inferences(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_inf_sample ON inferences(sample_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sel_status ON selection_queue(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_stage ON model_registry(stage)")

    def log_inference(
        self,
        sample_id: str,
        image_path: str,
        predicted_class: str,
        confidence: float,
        entropy: float,
        latency_ms: float,
        model_version: str,
        condition: str = "unknown",
        uploaded: int = 0
    ) -> int:
        """Insert inference record and return row ID."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO inferences (
                    sample_id, image_path, predicted_class, confidence, entropy,
                    latency_ms, model_version, condition, uploaded, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sample_id, image_path, predicted_class, confidence, entropy, latency_ms, model_version, condition, uploaded, now)
            )
            return cursor.lastrowid

    def log_drift(
        self,
        inference_id: Optional[int],
        features: Dict[str, float],
        is_drift: bool,
        triggered_features: List[str]
    ) -> int:
        """Insert drift metrics record."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO drift_records (
                    inference_id, brightness, contrast, edge_strength,
                    high_freq_std, high_freq_mean_abs, is_drift,
                    triggered_features, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    inference_id,
                    features.get("brightness", 0.0),
                    features.get("contrast", 0.0),
                    features.get("edge_strength", 0.0),
                    features.get("high_frequency_std", 0.0),
                    features.get("high_frequency_mean_abs", 0.0),
                    1 if is_drift else 0,
                    json.dumps(triggered_features),
                    now
                )
            )
            return cursor.lastrowid

    def add_to_selection_queue(
        self,
        inference_id: int,
        sample_id: str,
        image_path: str,
        strategy: str,
        upload_reason: str,
        entropy: float
    ) -> int:
        """Add record to selection queue for cloud upload / human review."""
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO selection_queue (
                    inference_id, sample_id, image_path, strategy, upload_reason, entropy, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (inference_id, sample_id, image_path, strategy, upload_reason, entropy, now)
            )
            # Mark inference as uploaded
            cursor.execute("UPDATE inferences SET uploaded = 1 WHERE id = ?", (inference_id,))
            return cursor.lastrowid

    def get_pending_review_items(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch pending review queue items with joined inference details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sq.id as queue_id, sq.inference_id, sq.sample_id, sq.image_path,
                       sq.strategy, sq.upload_reason, sq.entropy, sq.created_at,
                       inf.predicted_class, inf.confidence, inf.model_version, inf.condition
                FROM selection_queue sq
                JOIN inferences inf ON sq.inference_id = inf.id
                WHERE sq.status = 'pending'
                ORDER BY sq.entropy DESC
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def submit_review(
        self,
        queue_id: int,
        inference_id: int,
        sample_id: str,
        image_path: str,
        predicted_class: str,
        true_label: str,
        reviewer: str,
        notes: Optional[str] = None
    ) -> int:
        """Record human review decision and mark queue item reviewed."""
        now = datetime.now(timezone.utc).isoformat()
        is_corrected = 1 if predicted_class != true_label else 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO review_records (
                    inference_id, sample_id, image_path, predicted_class,
                    true_label, is_corrected, reviewer, notes, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (inference_id, sample_id, image_path, predicted_class, true_label, is_corrected, reviewer, notes, now)
            )
            review_id = cursor.lastrowid
            cursor.execute("UPDATE selection_queue SET status = 'reviewed' WHERE id = ?", (queue_id,))
            return review_id

    def get_model_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE version = ?", (version,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_champion_model(self) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE stage = 'champion' ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def register_model(
        self,
        version: str,
        model_name: str,
        checkpoint_path: str,
        stage: str,
        accuracy: Optional[float] = None,
        macro_f1: Optional[float] = None,
        trash_f1: Optional[float] = None,
        metrics_json: Optional[Dict[str, Any]] = None,
        approved_by: Optional[str] = None,
        promotion_reason: Optional[str] = None
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        metrics_str = json.dumps(metrics_json) if metrics_json else None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_registry (
                    version, model_name, checkpoint_path, stage,
                    accuracy, macro_f1, trash_f1, metrics_json,
                    approved_by, promotion_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    stage = excluded.stage,
                    accuracy = excluded.accuracy,
                    macro_f1 = excluded.macro_f1,
                    trash_f1 = excluded.trash_f1,
                    metrics_json = excluded.metrics_json,
                    approved_by = excluded.approved_by,
                    promotion_reason = excluded.promotion_reason,
                    updated_at = excluded.updated_at
                """,
                (version, model_name, checkpoint_path, stage, accuracy, macro_f1, trash_f1, metrics_str, approved_by, promotion_reason, now, now)
            )
            return cursor.lastrowid


# Global database accessor
_db_instance: Optional[Database] = None


def get_db(db_path: Optional[Path] = None) -> Database:
    global _db_instance
    if _db_instance is None or db_path is not None:
        _db_instance = Database(db_path)
    return _db_instance
