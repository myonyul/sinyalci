"""
Risk yönetimi modülü — Vadeli işlem (Futures).

LONG ve SHORT pozisyonlar için ATR tabanlı zarar durdur
ve kar hedefi seviyelerini hesaplar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

from .strategy_engine import SignalResult, SignalType

_INTERVAL_HOURS: dict[str, float] = {
    "1s": 1.0 / 3600.0,
    "1m": 1.0 / 60.0,
    "3m": 3.0 / 60.0,
    "5m": 5.0 / 60.0,
    "15m": 0.25,
    "30m": 0.5,
    "1h": 1.0,
    "2h": 2.0,
    "4h": 4.0,
    "6h": 6.0,
    "8h": 8.0,
    "12h": 12.0,
    "1d": 24.0,
    "3d": 72.0,
    "1w": 168.0,
    "1M": 720.0,
}

PositionType = Literal["LONG", "SHORT"]
SignalInput = Union[str, SignalType, PositionType]


@dataclass(frozen=True)
class RiskConfig:
    """ATR çarpanları ile risk parametreleri."""

    stop_loss_atr_mult: float = 1.5
    take_profit_1_atr_mult: float = 2.0
    take_profit_2_atr_mult: float = 4.0


@dataclass(frozen=True)
class ExitLevels:
    """Hesaplanmış çıkış seviyeleri."""

    entry_price: float
    signal_type: PositionType
    atr: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    config: RiskConfig

    @property
    def stop_loss_distance_pct(self) -> float:
        """Zarar durdurun girişe göre yüzde uzaklığı."""
        return ((self.stop_loss - self.entry_price) / self.entry_price) * 100

    @property
    def take_profit_1_distance_pct(self) -> float:
        """Birinci kar hedefinin girişe göre yüzde uzaklığı."""
        return ((self.take_profit_1 - self.entry_price) / self.entry_price) * 100

    @property
    def take_profit_2_distance_pct(self) -> float:
        """İkinci kar hedefinin girişe göre yüzde uzaklığı."""
        return ((self.take_profit_2 - self.entry_price) / self.entry_price) * 100


@dataclass(frozen=True)
class TargetEta:
    """Kar hedefine tahmini ulaşma süresi (ATR / mum)."""

    candles: float
    hours: float
    candle_text: str
    duration_text: str
    text: str


def interval_to_hours(interval: str) -> float:
    """Binance zaman dilimini saat cinsine çevirir."""
    text = (interval or "1h").strip()
    return _INTERVAL_HOURS.get(text, _INTERVAL_HOURS.get(text.lower(), 1.0))


def _format_hours(hours: float) -> str:
    """Saati okunabilir Türkçe süre metnine çevirir."""
    if hours < 1:
        minutes = max(1, int(round(hours * 60)))
        return f"~{minutes} dk"
    if hours < 24:
        if hours < 10:
            return f"~{hours:.1f} saat"
        return f"~{int(round(hours))} saat"
    days = hours / 24.0
    if days < 10:
        return f"~{days:.1f} gün"
    return f"~{int(round(days))} gün"


def _format_candles(candles: float) -> str:
    """Ortalama mum sayısını kısa metne çevirir."""
    if candles < 1:
        return "<1 mum"
    if candles >= 200:
        return "200+ mum"
    if candles < 10:
        return f"~{candles:.1f} mum"
    return f"~{int(round(candles))} mum"


def estimate_target_eta(
    entry_price: float,
    target_price: float,
    atr_value: float,
    interval: str = "1h",
) -> Optional[TargetEta]:
    """
    ATR ve zaman dilimine göre hedefe tahmini ulaşma süresini hesaplar.

    Ortalama mum sayısı = |hedef − giriş| / ATR.
    Süre (saat) = mum sayısı × dilim süresi.
    """
    try:
        entry = float(entry_price)
        target = float(target_price)
        atr = float(atr_value)
    except (TypeError, ValueError):
        return None

    if entry <= 0 or atr <= 0:
        return None

    distance = abs(target - entry)
    if distance <= 0:
        return None

    candles = distance / atr
    hours = candles * interval_to_hours(interval)
    candle_text = _format_candles(candles)
    duration_text = _format_hours(hours)
    return TargetEta(
        candles=candles,
        hours=hours,
        candle_text=candle_text,
        duration_text=duration_text,
        text=f"{candle_text} · {duration_text}",
    )


def _normalize_signal_type(signal_type: SignalInput) -> PositionType:
    """Sinyal tipini LONG veya SHORT olarak normalize eder."""
    if isinstance(signal_type, SignalType):
        if signal_type in (SignalType.LONG, SignalType.BUY):
            return "LONG"
        if signal_type in (SignalType.SHORT, SignalType.SELL):
            return "SHORT"
        raise ValueError(f"BEKLE sinyali için çıkış seviyesi hesaplanamaz: {signal_type}")

    normalized = str(signal_type).upper().strip()
    if normalized in ("LONG", "BUY", "AL"):
        return "LONG"
    if normalized in ("SHORT", "SELL", "SAT"):
        return "SHORT"

    raise ValueError(
        f"Geçersiz sinyal tipi: '{signal_type}'. Yalnızca LONG veya SHORT kabul edilir."
    )


def calculate_exit_levels(
    entry_price: float,
    signal_type: SignalInput,
    atr_value: float,
    config: RiskConfig | None = None,
) -> ExitLevels:
    """
    Giriş fiyatı, sinyal yönü ve ATR değerine göre çıkış seviyelerini hesaplar.

    LONG:
        - Zarar Durdur : Giriş − (1.5 × ATR)
        - Kar Hedefi 1 : Giriş + (2.0 × ATR)
        - Kar Hedefi 2 : Giriş + (4.0 × ATR)

    SHORT:
        - Zarar Durdur : Giriş + (1.5 × ATR)
        - Kar Hedefi 1 : Giriş − (2.0 × ATR)
        - Kar Hedefi 2 : Giriş − (4.0 × ATR)

    Parameters
    ----------
    entry_price : float
        Pozisyon giriş fiyatı.
    signal_type : str | SignalType
        ``LONG`` veya ``SHORT``.
    atr_value : float
        Strateji motorundan gelen ATR(14) değeri.
    config : RiskConfig, optional
        ATR çarpanları.

    Returns
    -------
    ExitLevels
        Hesaplanmış zarar durdur ve kar hedefi seviyeleri.

    Raises
    ------
    ValueError
        Geçersiz giriş fiyatı, ATR veya sinyal tipi.
    """
    if entry_price <= 0:
        raise ValueError(f"Giriş fiyatı pozitif olmalıdır, alınan: {entry_price}")
    if atr_value <= 0:
        raise ValueError(f"ATR değeri pozitif olmalıdır, alınan: {atr_value}")

    cfg = config or RiskConfig()
    direction = _normalize_signal_type(signal_type)

    sl_dist = cfg.stop_loss_atr_mult * atr_value
    tp1_dist = cfg.take_profit_1_atr_mult * atr_value
    tp2_dist = cfg.take_profit_2_atr_mult * atr_value

    if direction == "LONG":
        stop_loss = entry_price - sl_dist
        take_profit_1 = entry_price + tp1_dist
        take_profit_2 = entry_price + tp2_dist
    else:
        stop_loss = entry_price + sl_dist
        take_profit_1 = entry_price - tp1_dist
        take_profit_2 = entry_price - tp2_dist

    return ExitLevels(
        entry_price=entry_price,
        signal_type=direction,
        atr=atr_value,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        config=cfg,
    )


def calculate_exit_levels_from_signal(
    signal_result: SignalResult,
    config: RiskConfig | None = None,
) -> ExitLevels | None:
    """
    Strateji sinyal sonucundan çıkış seviyelerini hesaplar.

    LONG veya SHORT sinyallerinde seviye döndürür; BEKLE'de ``None``.

    Parameters
    ----------
    signal_result : SignalResult
        strategy_engine.evaluate_signal() çıktısı (ATR içermeli).
    config : RiskConfig, optional
        ATR çarpanları.

    Returns
    -------
    ExitLevels or None
        Aktif sinyal varsa hesaplanmış seviyeler.
    """
    label = signal_result.signal_label
    if label == "BEKLE" or signal_result.signal in (SignalType.BEKLE, SignalType.HOLD):
        return None

    if signal_result.close is None or signal_result.close <= 0:
        raise ValueError("Geçerli bir giriş (kapanış) fiyatı gerekli.")
    if signal_result.atr is None or signal_result.atr <= 0:
        raise ValueError("Geçerli bir ATR değeri gerekli — strateji motorundan alınamadı.")

    signal_type: SignalInput = label
    if label == "BEKLE":
        signal_type = signal_result.signal

    return calculate_exit_levels(
        entry_price=float(signal_result.close),
        signal_type=signal_type,
        atr_value=float(signal_result.atr),
        config=config,
    )


def _eta_fields(
    entry_price: float,
    target_price: float,
    atr_value: float,
    interval: str,
) -> dict[str, Any]:
    """Kar hedefi sözlüğüne eklenecek tahmini süre alanları."""
    eta = estimate_target_eta(entry_price, target_price, atr_value, interval)
    if eta is None:
        return {"eta_text": None, "eta_candles": None, "eta_hours": None}
    return {
        "eta_text": eta.text,
        "eta_candles": round(eta.candles, 2),
        "eta_hours": round(eta.hours, 2),
    }


def to_json_dict(
    levels: ExitLevels,
    symbol: str = "",
    price_precision: int = 4,
    interval: str = "1h",
) -> dict[str, Any]:
    """
    Çıkış seviyelerini JSON uyumlu sözlük olarak döndürür.

    Parameters
    ----------
    levels : ExitLevels
        calculate_exit_levels() çıktısı.
    symbol : str, optional
        Parite adı.
    price_precision : int, optional
        Fiyat yuvarlama basamağı.
    interval : str, optional
        Mum zaman dilimi — tahmini hedef süresi için.

    Returns
    -------
    dict
        JSON serileştirilebilir temiz sözlük.
    """

    def _round(value: float) -> float:
        return round(value, price_precision)

    cfg = levels.config

    return {
        "signal": levels.signal_type,
        "symbol": symbol.upper() if symbol else None,
        "entry_price": _round(levels.entry_price),
        "atr": _round(levels.atr),
        "interval": interval,
        "stop_loss": {
            "price": _round(levels.stop_loss),
            "atr_multiple": cfg.stop_loss_atr_mult,
            "distance_pct": round(levels.stop_loss_distance_pct, 4),
            "label": "Zarar Durdur",
        },
        "take_profit_1": {
            "price": _round(levels.take_profit_1),
            "atr_multiple": cfg.take_profit_1_atr_mult,
            "distance_pct": round(levels.take_profit_1_distance_pct, 4),
            "label": "Kar Hedefi 1",
            **_eta_fields(
                levels.entry_price,
                levels.take_profit_1,
                levels.atr,
                interval,
            ),
        },
        "take_profit_2": {
            "price": _round(levels.take_profit_2),
            "atr_multiple": cfg.take_profit_2_atr_mult,
            "distance_pct": round(levels.take_profit_2_distance_pct, 4),
            "label": "Kar Hedefi 2",
            **_eta_fields(
                levels.entry_price,
                levels.take_profit_2,
                levels.atr,
                interval,
            ),
        },
        "risk_config": {
            "stop_loss_atr": cfg.stop_loss_atr_mult,
            "take_profit_1_atr": cfg.take_profit_1_atr_mult,
            "take_profit_2_atr": cfg.take_profit_2_atr_mult,
        },
    }


def build_risk_payload(
    entry_price: float,
    signal_type: SignalInput,
    atr_value: float,
    symbol: str = "",
    config: RiskConfig | None = None,
    price_precision: int = 4,
    interval: str = "1h",
) -> dict[str, Any]:
    """
    Giriş, sinyal ve ATR parametrelerinden JSON uyumlu risk sözlüğü üretir.

    Parameters
    ----------
    entry_price : float
        Pozisyon giriş fiyatı.
    signal_type : str | SignalType
        LONG veya SHORT.
    atr_value : float
        ATR(14) değeri.
    symbol : str, optional
        Parite adı.
    config : RiskConfig, optional
        ATR çarpanları.
    price_precision : int, optional
        Fiyat yuvarlama basamağı.

    Returns
    -------
    dict
        JSON uyumlu risk seviyeleri sözlüğü.
    """
    levels = calculate_exit_levels(
        entry_price=entry_price,
        signal_type=signal_type,
        atr_value=atr_value,
        config=config,
    )
    payload = to_json_dict(
        levels,
        symbol=symbol,
        price_precision=price_precision,
        interval=interval,
    )
    if payload.get("symbol") is None:
        payload.pop("symbol", None)
    return payload


def build_risk_payload_from_signal(
    signal_result: SignalResult,
    symbol: str = "",
    config: RiskConfig | None = None,
    price_precision: int = 4,
    interval: str = "1h",
) -> dict[str, Any] | None:
    """
    Strateji sinyal sonucundan JSON uyumlu risk sözlüğü üretir.

    LONG veya SHORT sinyallerinde dolu sözlük; BEKLE'de ``None``.

    Parameters
    ----------
    signal_result : SignalResult
        Strateji değerlendirme sonucu.
    symbol : str, optional
        Parite adı.
    config : RiskConfig, optional
        ATR çarpanları.
    price_precision : int, optional
        Fiyat yuvarlama basamağı.

    Returns
    -------
    dict or None
        Risk seviyeleri sözlüğü veya BEKLE sinyalinde None.
    """
    levels = calculate_exit_levels_from_signal(signal_result, config=config)
    if levels is None:
        return None

    return to_json_dict(
        levels,
        symbol=symbol,
        price_precision=price_precision,
        interval=interval,
    )
