"""
Smart Notes Assistant - Storage Module
Persistent SQLite storage for notes with JSON export/import support,
CRUD operations, category statistics, and strict duplicate prevention.
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = "data/notes.db"

SAMPLE_NOTES = [
    {
        "text": "My Python exam is on Friday at 10 AM.",
        "category": "Education",
        "confidence": 0.94,
        "is_uncertain": False,
    },
    {
        "text": "Pay the electricity bill before the 28th.",
        "category": "Finance",
        "confidence": 0.96,
        "is_uncertain": False,
    },
    {
        "text": "Flight to Delhi is booked for Monday morning.",
        "category": "Travel",
        "confidence": 0.93,
        "is_uncertain": False,
    },
    {
        "text": "Buy groceries this weekend.",
        "category": "Shopping",
        "confidence": 0.91,
        "is_uncertain": False,
    },
    {
        "text": "Call grandma this weekend.",
        "category": "Personal",
        "confidence": 0.52,
        "is_uncertain": True,
    },
    {
        "text": "Prepare quarterly presentation for Monday client meeting.",
        "category": "Work",
        "confidence": 0.92,
        "is_uncertain": False,
    },
    {
        "text": "Dentist checkup appointment on Thursday 4 PM.",
        "category": "Health",
        "confidence": 0.95,
        "is_uncertain": False,
    },
    {
        "text": "Alex's birthday party on Saturday evening at 7 PM.",
        "category": "Events",
        "confidence": 0.93,
        "is_uncertain": False,
    },
]


class NotesStorage:
    """
    SQLite-backed persistent notes manager.
    Guarantees persistence across application restarts.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create notes schema if not exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    is_uncertain INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def add_note(
        self,
        text: str,
        category: str,
        confidence: float = 0.0,
        is_uncertain: bool = False,
        created_at: Optional[str] = None,
    ) -> int:
        """Add a new note to the database."""
        now = created_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO notes (text, category, confidence, is_uncertain, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (text.strip(), category, round(float(confidence), 4), int(is_uncertain), now, now),
            )
            conn.commit()
            return cursor.lastrowid

    def get_all_notes(
        self,
        category_filter: Optional[str] = None,
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Retrieve notes with optional category filter and sorting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            order = "DESC" if sort_order.lower() == "desc" else "ASC"
            if category_filter and category_filter != "All":
                cursor.execute(
                    f"SELECT * FROM notes WHERE category = ? ORDER BY id {order}",
                    (category_filter,),
                )
            else:
                cursor.execute(f"SELECT * FROM notes ORDER BY id {order}")
            rows = cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "category": r["category"],
                    "confidence": float(r["confidence"]),
                    "is_uncertain": bool(r["is_uncertain"]),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    def get_note_by_id(self, note_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single note by primary key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return {
                "id": r["id"],
                "text": r["text"],
                "category": r["category"],
                "confidence": float(r["confidence"]),
                "is_uncertain": bool(r["is_uncertain"]),
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }

    def update_note(
        self,
        note_id: int,
        text: Optional[str] = None,
        category: Optional[str] = None,
        confidence: Optional[float] = None,
        is_uncertain: Optional[bool] = None,
    ) -> bool:
        """Update an existing note's content or assigned category."""
        existing = self.get_note_by_id(note_id)
        if not existing:
            return False

        new_text = text.strip() if text is not None else existing["text"]
        new_category = category if category is not None else existing["category"]
        new_confidence = (
            round(float(confidence), 4) if confidence is not None else existing["confidence"]
        )
        new_uncertain = (
            int(is_uncertain) if is_uncertain is not None else int(existing["is_uncertain"])
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE notes
                SET text = ?, category = ?, confidence = ?, is_uncertain = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_text, new_category, new_confidence, new_uncertain, now, note_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_note(self, note_id: int) -> bool:
        """Delete a note by primary key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_note_count(self) -> int:
        """Return total count of stored notes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS count FROM notes")
            row = cursor.fetchone()
            return row["count"] if row else 0

    def get_category_distribution(self) -> Dict[str, int]:
        """Return count of notes grouped by category."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT category, COUNT(*) as count FROM notes GROUP BY category ORDER BY count DESC"
            )
            rows = cursor.fetchall()
            return {r["category"]: r["count"] for r in rows}

    def load_sample_notes(
        self, samples: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Load sample notes into storage while strictly preventing duplicate notes.
        Matches by normalized text content (case-insensitive and trimmed).
        Returns the number of newly added notes.
        """
        sample_list = samples if samples is not None else SAMPLE_NOTES
        existing_notes = self.get_all_notes()
        existing_texts = {
            n["text"].strip().lower() for n in existing_notes
        }

        added_count = 0
        for sample in sample_list:
            clean_text = sample["text"].strip().lower()
            if clean_text not in existing_texts:
                self.add_note(
                    text=sample["text"],
                    category=sample["category"],
                    confidence=sample.get("confidence", 0.9),
                    is_uncertain=sample.get("is_uncertain", False),
                )
                existing_texts.add(clean_text)
                added_count += 1

        return added_count

    def clear_all_notes(self) -> int:
        """Clear all stored notes (useful for testing and reset)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM notes")
            conn.commit()
            return cursor.rowcount

    def export_to_json(self, json_path: str) -> int:
        """Export all notes to a JSON file."""
        notes = self.get_all_notes()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        return len(notes)

    def import_from_json(self, json_path: str) -> int:
        """Import notes from a JSON file, skipping duplicates."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return self.load_sample_notes(data)
        return 0
