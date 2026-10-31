"""Dynamic waste class registry and bin mapping manager for EdgeLoop.

Maintains persistence of standard TrashNet classes and dynamic user-created
classes across database, JSON configurations, and restarts.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from edgeloop.common.config import get_config
from edgeloop.common.database import get_db

logger = logging.getLogger("edgeloop.classes")

# Standard 6 TrashNet Baseline Classes
DEFAULT_CLASSES: List[Dict[str, Any]] = [
    {
        "class_name": "cardboard",
        "display_name": "Cardboard",
        "category": "Paper & Stiff Packaging",
        "bin_name": "Brown/Blue Bin (Cardboard)",
        "bin_code": "CARDBOARD_BROWN",
        "bin_guidance": "Flattened corrugated boxes, carton packaging, and stiff book covers.",
        "color_rgb": "141,110,99",
        "color_bgr": "99,110,141"
    },
    {
        "class_name": "glass",
        "display_name": "Glass",
        "category": "Beverage & Food Containers",
        "bin_name": "Green Bin (Glass)",
        "bin_code": "GLASS_GREEN",
        "bin_guidance": "Glass bottles, jars, and beverage glassware (rinse first).",
        "color_rgb": "67,160,71",
        "color_bgr": "71,160,67"
    },
    {
        "class_name": "metal",
        "display_name": "Metal",
        "category": "Cans & Foils",
        "bin_name": "Grey/Yellow Bin (Metals)",
        "bin_code": "METAL_GREY",
        "bin_guidance": "Aluminum beverage cans, tin food cans, and clean metal foil.",
        "color_rgb": "120,144,156",
        "color_bgr": "156,144,120"
    },
    {
        "class_name": "paper",
        "display_name": "Paper",
        "category": "Office Paper & Newsprint",
        "bin_name": "Blue Bin (Paper & Books)",
        "bin_code": "PAPER_BLUE",
        "bin_guidance": "Clean paper, books, notebooks, envelopes, and newsprint.",
        "color_rgb": "30,136,229",
        "color_bgr": "229,136,30"
    },
    {
        "class_name": "plastic",
        "display_name": "Plastic",
        "category": "Rigid & Film Plastics",
        "bin_name": "Yellow Bin (Plastics)",
        "bin_code": "PLASTIC_YELLOW",
        "bin_guidance": "Plastic bottles, drink cups, jugs, and clean rigid plastic containers.",
        "color_rgb": "253,216,53",
        "color_bgr": "53,216,253"
    },
    {
        "class_name": "trash",
        "display_name": "Trash",
        "category": "Non-Recyclable General Refuse",
        "bin_name": "Black Bin (General Refuse)",
        "bin_code": "TRASH_BLACK",
        "bin_guidance": "Composite packaging, soiled items, and non-recyclable solid waste.",
        "color_rgb": "220,53,69",
        "color_bgr": "69,53,220"
    }
]


def normalize_class_name(name: str) -> str:
    """Normalize user input to clean snake_case identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return cleaned or "custom_object"


def ensure_classes_table() -> None:
    """Ensure the waste_classes table exists and contains baseline classes."""
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS waste_classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                category TEXT NOT NULL,
                bin_name TEXT NOT NULL,
                bin_code TEXT NOT NULL,
                bin_guidance TEXT NOT NULL,
                color_rgb TEXT NOT NULL,
                color_bgr TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_class_name ON waste_classes(class_name)")

        # Check if baseline is seeded
        cursor.execute("SELECT COUNT(*) as cnt FROM waste_classes")
        row = cursor.fetchone()
        if row is None or row["cnt"] == 0:
            now = datetime.now(timezone.utc).isoformat()
            for cls_info in DEFAULT_CLASSES:
                cursor.execute("""
                    INSERT OR IGNORE INTO waste_classes (
                        class_name, display_name, category, bin_name, bin_code,
                        bin_guidance, color_rgb, color_bgr, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cls_info["class_name"],
                    cls_info["display_name"],
                    cls_info["category"],
                    cls_info["bin_name"],
                    cls_info["bin_code"],
                    cls_info["bin_guidance"],
                    cls_info["color_rgb"],
                    cls_info["color_bgr"],
                    now
                ))


def get_registered_classes() -> List[Dict[str, Any]]:
    """Retrieve all registered classes from the database."""
    ensure_classes_table()
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM waste_classes ORDER BY id ASC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_class_names() -> List[str]:
    """Retrieve ordered list of class name identifiers."""
    classes = get_registered_classes()
    return [c["class_name"] for c in classes]


def register_new_class(
    display_name: str,
    category: str,
    bin_name: str,
    bin_code: Optional[str] = None,
    bin_guidance: Optional[str] = None,
    color_rgb: Optional[str] = None,
    color_bgr: Optional[str] = None
) -> Dict[str, Any]:
    """
    Register a dynamic new waste class.
    Persists to SQLite database and synchronizes class_mapping.json.
    """
    ensure_classes_table()
    class_name = normalize_class_name(display_name)
    disp_name = display_name.strip()
    cat = category.strip() or "Custom Recyclable"
    b_name = bin_name.strip() or "Yellow Bin (Recyclables)"
    b_code = bin_code.strip() if bin_code else f"{class_name.upper()}_BIN"
    b_guidance = bin_guidance.strip() if bin_guidance else f"Recyclable {disp_name} item. Clean before disposal."

    # Default colors (distinct orange/amber for new items if not provided)
    c_rgb = color_rgb or "255,140,0"
    c_bgr = color_bgr or "0,140,255"

    now = datetime.now(timezone.utc).isoformat()
    db = get_db()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO waste_classes (
                class_name, display_name, category, bin_name, bin_code,
                bin_guidance, color_rgb, color_bgr, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(class_name) DO UPDATE SET
                display_name=excluded.display_name,
                category=excluded.category,
                bin_name=excluded.bin_name,
                bin_code=excluded.bin_code,
                bin_guidance=excluded.bin_guidance,
                color_rgb=excluded.color_rgb,
                color_bgr=excluded.color_bgr
        """, (class_name, disp_name, cat, b_name, b_code, b_guidance, c_rgb, c_bgr, now))

    sync_class_mapping_json()
    logger.info("Registered dynamic class '%s' (%s) -> %s", class_name, disp_name, b_name)

    return {
        "class_name": class_name,
        "display_name": disp_name,
        "category": cat,
        "bin_name": b_name,
        "bin_code": b_code,
        "bin_guidance": b_guidance,
        "color_rgb": c_rgb,
        "color_bgr": c_bgr,
        "created_at": now
    }


def sync_class_mapping_json() -> None:
    """Synchronize registered classes to configs/class_mapping.json."""
    class_names = get_class_names()
    config = get_config()
    mapping_path = Path(config.root_dir) / "configs" / "class_mapping.json"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "classes": class_names,
        "class_to_idx": {c: i for i, c in enumerate(class_names)},
        "idx_to_class": {str(i): c for i, c in enumerate(class_names)}
    }
    with open(mapping_path, "w") as f:
        json.dump(data, f, indent=2)


def get_bin_mappings() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of bin mappings for all registered classes."""
    classes = get_registered_classes()
    mapping: Dict[str, Dict[str, Any]] = {}
    for c in classes:
        rgb_parts = [int(p) for p in c["color_rgb"].split(",")]
        bgr_parts = [int(p) for p in c["color_bgr"].split(",")]
        mapping[c["class_name"]] = {
            "bin_name": c["bin_name"],
            "bin_code": c["bin_code"],
            "color_rgb": tuple(rgb_parts),
            "color_bgr": tuple(bgr_parts),
            "guidance": c["bin_guidance"],
            "display_name": c["display_name"],
            "category": c["category"]
        }
    return mapping


def get_bin_mapping(class_name: str) -> Dict[str, Any]:
    """Retrieve bin mapping for a specific class with fallback to trash."""
    mappings = get_bin_mappings()
    if class_name in mappings:
        return mappings[class_name]

    # Fallback to general refuse (trash)
    return mappings.get("trash", {
        "bin_name": "Black Bin (General Refuse)",
        "bin_code": "TRASH_BLACK",
        "color_rgb": (220, 53, 69),
        "color_bgr": (69, 53, 220),
        "guidance": "Composite packaging, soiled items, and non-recyclable solid waste.",
        "display_name": class_name.replace("_", " ").title(),
        "category": "General Refuse"
    })
