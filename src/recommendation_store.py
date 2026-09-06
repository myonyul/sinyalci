"""
Tavsiye geçmişi kalıcı depolama.

Piyasa radarı sinyalleri ``data/recommendations.json`` dosyasına yazılır;
Streamlit oturumu kapansa bile son kayıtlar korunur.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_LIMIT = 5
_HISTORY_FILENAME = "recommendations.json"


def _project_root() -> Path:
    """Proje kök dizinini döndürür."""
    return Path(__file__).resolve().parent.parent


def get_history_file_path() -> Path:
    """Tavsiye geçmişi JSON dosya yolunu döndürür."""
    return _project_root() / "data" / _HISTORY_FILENAME


def load_recommendation_history(
    max_items: int = DEFAULT_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """
    Diskten tavsiye geçmişini okur.

    Dosya yoksa veya bozuksa boş liste döner.
    """
    path = get_history_file_path()
    if not path.is_file():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    history: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and item.get("symbol"):
            history.append(item)
        if len(history) >= max_items:
            break
    return history


def save_recommendation_history(
    history: list[dict[str, Any]],
    max_items: int = DEFAULT_HISTORY_LIMIT,
) -> None:
    """Tavsiye geçmişini diske yazar."""
    path = get_history_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    trimmed = history[:max_items]
    payload = json.dumps(trimmed, ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


def clear_recommendation_history() -> None:
    """Tavsiye geçmişi dosyasını temizler."""
    save_recommendation_history([])
