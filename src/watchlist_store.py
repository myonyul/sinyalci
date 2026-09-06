"""
İzleme listesi pozisyon depolama.

Kullanıcının girdiği alış fiyatı ve kar hedefi ``data/watchlist.json``
dosyasında kalıcı olarak saklanır.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_WATCHLIST_FILENAME = "watchlist.json"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_watchlist_file_path() -> Path:
    """İzleme listesi JSON dosya yolunu döndürür."""
    return _project_root() / "data" / _WATCHLIST_FILENAME


def load_watchlist_positions() -> dict[str, dict[str, Any]]:
    """Diskten izleme listesi pozisyonlarını okur."""
    path = get_watchlist_file_path()
    if not path.is_file():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    positions: dict[str, dict[str, Any]] = {}
    for symbol, payload in data.items():
        if isinstance(payload, dict) and symbol:
            positions[str(symbol).upper()] = payload
    return positions


def save_watchlist_positions(positions: dict[str, dict[str, Any]]) -> None:
    """İzleme listesi pozisyonlarını diske yazar."""
    path = get_watchlist_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(positions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_watchlist_position(
    symbol: str,
    entry_price: float,
    target_price: float,
    stop_loss: Optional[float] = None,
    signal_label: str = "LONG",
) -> dict[str, Any]:
    """Tek parite için pozisyon kaydeder ve güncel sözlüğü döndürür."""
    symbol = symbol.upper()
    positions = load_watchlist_positions()
    positions[symbol] = {
        "symbol": symbol,
        "entry_price": float(entry_price),
        "target_price": float(target_price),
        "stop_loss": float(stop_loss) if stop_loss is not None else None,
        "signal_label": signal_label,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    save_watchlist_positions(positions)
    return positions


def remove_watchlist_position(symbol: str) -> dict[str, dict[str, Any]]:
    """Parite pozisyon kaydını siler."""
    symbol = symbol.upper()
    positions = load_watchlist_positions()
    positions.pop(symbol, None)
    save_watchlist_positions(positions)
    return positions
