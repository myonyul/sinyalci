"""
Vadeli işlem (Futures) strateji motoru.

OHLCV verisine EMA, RSI, MACD ve ATR indikatörlerini ekler;
LONG / SHORT / BEKLE sinyali ve detaylı Türkçe gerekçe üretir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import sys
from typing import Any, Literal, Optional

import pandas as pd
import pandas_ta as ta

SignalLabel = Literal["LONG", "SHORT", "BEKLE"]


# ---------------------------------------------------------------------------
# Sinyal tipleri ve yapılandırma
# ---------------------------------------------------------------------------


class SignalType(str, Enum):
    """Strateji çıktısı olarak kullanılan sinyal türleri."""

    LONG = "🟢 LONG (UZUN)"
    SHORT = "🔴 SHORT (KISA)"
    BEKLE = "⚪ BEKLE"

    # Geriye dönük uyumluluk (spot / eski modüller)
    BUY = "🟢 LONG (UZUN)"
    SELL = "🔴 SHORT (KISA)"
    HOLD = "⚪ BEKLE"


@dataclass(frozen=True)
class FuturesStrategyConfig:
    """Vadeli işlem trend stratejisi parametreleri."""

    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    rsi_long_min: float = 50.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 50.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    severe_drop_pct: float = -10.0
    strong_rise_pct: float = 10.0
    early_long_rsi_max: float = 25.0
    oversold_rsi_context: float = 30.0


@dataclass(frozen=True)
class MarketContext:
    """24 saatlik piyasa bağlamı (Binance get_ticker verisi)."""

    price_change_pct_24h: Optional[float] = None

    @classmethod
    def from_ticker(cls, price_change_percent: Optional[float]) -> MarketContext:
        """get_ticker priceChangePercent değerinden bağlam oluşturur."""
        if price_change_percent is None:
            return cls()
        return cls(price_change_pct_24h=float(price_change_percent))


@dataclass
class IndicatorSnapshot:
    """Son mumdaki indikatör anlık görüntüsü."""

    close: float
    ema_20: float
    ema_50: float
    ema_200: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    atr: float
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class SignalResult:
    """Strateji değerlendirmesinin detaylı sonucu."""

    signal: SignalType
    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    timestamp: Optional[pd.Timestamp]
    close: Optional[float]
    atr: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    message: str = ""
    signal_label: SignalLabel = "BEKLE"
    is_early_long: bool = False
    price_change_pct_24h: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Sinyali sözlük formatında döndürür."""
        return {
            "signal": self.signal_label,
            "reason": self.message,
            "atr": self.atr,
        }


# Geriye dönük uyumluluk
StrategyConfig = FuturesStrategyConfig


# ---------------------------------------------------------------------------
# Doğrulama ve indikatör hesaplama
# ---------------------------------------------------------------------------


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Girdi DataFrame'inin strateji için uygun olduğunu doğrular."""
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame'de eksik sütunlar: {sorted(missing)}")
    if df.empty:
        raise ValueError("DataFrame boş — strateji çalıştırılamaz.")
    if len(df) < 200:
        raise ValueError(
            f"EMA 200 hesabı için en az 200 mum gerekir, mevcut: {len(df)}"
        )


def _macd_columns(config: FuturesStrategyConfig) -> tuple[str, str, str]:
    """pandas-ta MACD sütun adlarını döndürür."""
    suffix = f"{config.macd_fast}_{config.macd_slow}_{config.macd_signal}"
    return (
        f"MACD_{suffix}",
        f"MACDs_{suffix}",
        f"MACDh_{suffix}",
    )


def add_indicators(
    df: pd.DataFrame,
    config: FuturesStrategyConfig | None = None,
) -> pd.DataFrame:
    """
    Vadeli işlem stratejisi için tüm indikatörleri DataFrame'e ekler.

    Eklenen sütunlar: EMA_20, EMA_50, EMA_200, RSI_14, MACD, ATR_14
    """
    _validate_ohlcv(df)
    cfg = config or FuturesStrategyConfig()
    result = df.copy()

    result[f"EMA_{cfg.ema_fast}"] = ta.ema(result["Close"], length=cfg.ema_fast)
    result[f"EMA_{cfg.ema_mid}"] = ta.ema(result["Close"], length=cfg.ema_mid)
    result[f"EMA_{cfg.ema_slow}"] = ta.ema(result["Close"], length=cfg.ema_slow)
    result[f"RSI_{cfg.rsi_period}"] = ta.rsi(result["Close"], length=cfg.rsi_period)

    macd_df = ta.macd(
        result["Close"],
        fast=cfg.macd_fast,
        slow=cfg.macd_slow,
        signal=cfg.macd_signal,
    )
    if macd_df is None:
        raise ValueError("MACD hesaplanamadı — yeterli veri olmayabilir.")
    result = pd.concat([result, macd_df], axis=1)

    result[f"ATR_{cfg.atr_period}"] = ta.atr(
        result["High"],
        result["Low"],
        result["Close"],
        length=cfg.atr_period,
    )

    return result


def _extract_snapshot(
    df: pd.DataFrame,
    config: FuturesStrategyConfig,
) -> IndicatorSnapshot:
    """Son mumdan indikatör değerlerini çıkarır."""
    cfg = config
    macd_col, macd_sig_col, macd_hist_col = _macd_columns(cfg)
    last = df.iloc[-1]

    def _float(col: str) -> float:
        value = last[col]
        if pd.isna(value):
            raise ValueError(f"{col} değeri hesaplanamadı (yetersiz veri).")
        return float(value)

    return IndicatorSnapshot(
        close=_float("Close"),
        ema_20=_float(f"EMA_{cfg.ema_fast}"),
        ema_50=_float(f"EMA_{cfg.ema_mid}"),
        ema_200=_float(f"EMA_{cfg.ema_slow}"),
        rsi=_float(f"RSI_{cfg.rsi_period}"),
        macd=_float(macd_col),
        macd_signal=_float(macd_sig_col),
        macd_histogram=_float(macd_hist_col),
        atr=_float(f"ATR_{cfg.atr_period}"),
        timestamp=df.index[-1],
    )


def _ema_bullish(df: pd.DataFrame, config: FuturesStrategyConfig) -> bool:
    """EMA 20, EMA 50'nin üzerinde veya yukarı kesmiş mi?"""
    fast_col = f"EMA_{config.ema_fast}"
    mid_col = f"EMA_{config.ema_mid}"

    last = df.iloc[-1]
    if last[fast_col] > last[mid_col]:
        return True

    if len(df) < 2:
        return False

    prev = df.iloc[-2]
    if pd.isna(prev[fast_col]) or pd.isna(prev[mid_col]):
        return False

    return prev[fast_col] <= prev[mid_col] and last[fast_col] > last[mid_col]


def _check_long_conditions(snap: IndicatorSnapshot, df: pd.DataFrame, cfg: FuturesStrategyConfig) -> tuple[bool, list[str]]:
    """LONG koşullarını kontrol eder; (uygun_mu, koşul_listesi) döner."""
    checks: list[str] = []

    trend_ok = snap.close > snap.ema_200
    checks.append(
        f"Genel trend (Kapanış > EMA 200): "
        f"{'✓' if trend_ok else '✗'} "
        f"({snap.close:.4f} {'>' if trend_ok else '<='} {snap.ema_200:.4f})"
    )

    ema_ok = _ema_bullish(df, cfg)
    checks.append(
        f"Kısa vadeli momentum (EMA 20 > EMA 50 veya yukarı kesişim): "
        f"{'✓' if ema_ok else '✗'} "
        f"(EMA 20: {snap.ema_20:.4f}, EMA 50: {snap.ema_50:.4f})"
    )

    rsi_ok = cfg.rsi_long_min <= snap.rsi <= cfg.rsi_long_max
    checks.append(
        f"RSI momentum bölgesi ({cfg.rsi_long_min:.0f}–{cfg.rsi_long_max:.0f}): "
        f"{'✓' if rsi_ok else '✗'} (RSI: {snap.rsi:.2f})"
    )

    macd_ok = snap.macd_histogram > 0
    checks.append(
        f"MACD histogramı pozitif: "
        f"{'✓' if macd_ok else '✗'} (Histogram: {snap.macd_histogram:.6f})"
    )

    return trend_ok and ema_ok and rsi_ok and macd_ok, checks


def _check_short_conditions(snap: IndicatorSnapshot, cfg: FuturesStrategyConfig) -> tuple[bool, list[str]]:
    """SHORT koşullarını kontrol eder; (uygun_mu, koşul_listesi) döner."""
    checks: list[str] = []

    trend_ok = snap.close < snap.ema_200
    checks.append(
        f"Genel trend (Kapanış < EMA 200): "
        f"{'✓' if trend_ok else '✗'} "
        f"({snap.close:.4f} {'<' if trend_ok else '>='} {snap.ema_200:.4f})"
    )

    ema_ok = snap.ema_20 < snap.ema_50
    checks.append(
        f"Kısa vadeli zayıflık (EMA 20 < EMA 50): "
        f"{'✓' if ema_ok else '✗'} "
        f"(EMA 20: {snap.ema_20:.4f}, EMA 50: {snap.ema_50:.4f})"
    )

    rsi_ok = cfg.rsi_short_min <= snap.rsi <= cfg.rsi_short_max
    checks.append(
        f"RSI düşüş momentumu ({cfg.rsi_short_min:.0f}–{cfg.rsi_short_max:.0f}): "
        f"{'✓' if rsi_ok else '✗'} (RSI: {snap.rsi:.2f})"
    )

    macd_ok = snap.macd_histogram < 0
    checks.append(
        f"MACD histogramı negatif: "
        f"{'✓' if macd_ok else '✗'} (Histogram: {snap.macd_histogram:.6f})"
    )

    return trend_ok and ema_ok and rsi_ok and macd_ok, checks


def _is_green_recovery_candle(df: pd.DataFrame) -> bool:
    """Son mum yeşil mi (kapanış > açılış) — toparlanma mumu kontrolü."""
    last = df.iloc[-1]
    return float(last["Close"]) > float(last["Open"])


def _check_early_long_conditions(
    snap: IndicatorSnapshot,
    df: pd.DataFrame,
    cfg: FuturesStrategyConfig,
    ctx: MarketContext | None,
) -> tuple[bool, list[str]]:
    """
    Riskli/Erken LONG koşulları — sert düşüş + derin aşırı satım + yeşil mum.

    MACD pozitif olmak zorunda değildir (gecikmeli indikatör).
    """
    checks: list[str] = []

    if ctx is None or ctx.price_change_pct_24h is None:
        checks.append("24s piyasa bağlamı: ✗ (veri yok)")
        return False, checks

    drop_ok = ctx.price_change_pct_24h <= cfg.severe_drop_pct
    checks.append(
        f"Sert 24s düşüş (<= %{cfg.severe_drop_pct:.0f}): "
        f"{'✓' if drop_ok else '✗'} "
        f"(Değişim: %{ctx.price_change_pct_24h:+.2f})"
    )

    rsi_ok = snap.rsi < cfg.early_long_rsi_max
    checks.append(
        f"Derin aşırı satım (RSI < {cfg.early_long_rsi_max:.0f}): "
        f"{'✓' if rsi_ok else '✗'} (RSI: {snap.rsi:.2f})"
    )

    green_ok = _is_green_recovery_candle(df)
    checks.append(
        f"Toparlanma mumu (yeşil mum): "
        f"{'✓' if green_ok else '✗'}"
    )

    checks.append(
        f"MACD histogramı (erken sinyal — zorunlu değil): "
        f"{'pozitif' if snap.macd_histogram > 0 else 'negatif/gecikmeli'} "
        f"({snap.macd_histogram:.6f})"
    )

    return drop_ok and rsi_ok and green_ok, checks


def _build_market_context_intro(
    ctx: MarketContext | None,
    snap: IndicatorSnapshot,
    signal_kind: Literal["LONG", "SHORT", "EARLY_LONG", "BEKLE"],
    cfg: FuturesStrategyConfig,
) -> str:
    """24s fiyat değişimine göre gerekçeye eklenecek piyasa bağlamı paragrafı."""
    if ctx is None or ctx.price_change_pct_24h is None:
        return ""

    pct = ctx.price_change_pct_24h

    if signal_kind in ("LONG", "EARLY_LONG"):
        if pct <= cfg.severe_drop_pct and snap.rsi < cfg.oversold_rsi_context:
            return (
                f"Bu coin son 24 saatte şiddetli satış yemiş ({pct:.1f}%). "
                f"RSI aşırı satım bölgesinde ({snap.rsi:.1f}) — "
                f"dipten dönüş / tepki alımı fırsatı. "
            )
        if pct >= cfg.strong_rise_pct and snap.ema_20 > snap.ema_50:
            return (
                f"Güçlü bir yükseliş trendi var ({pct:+.1f}% artış); "
                f"trende katılım fırsatı (Momentum LONG). "
            )
        if pct >= 5:
            return (
                f"Son 24 saatte yükseliş var ({pct:+.1f}%) — "
                f"kısa vadeli momentum lehine. "
            )
        if pct <= -5:
            return (
                f"Son 24 saatte düşüş baskısı var ({pct:.1f}%) — "
                f"dipten dönüş arayanlar için dikkatli takip gerekir. "
            )

    if signal_kind == "SHORT":
        if pct >= cfg.strong_rise_pct:
            return (
                f"Coin son 24 saatte belirgin yükselmiş ({pct:+.1f}%); "
                f"aşırı alım / kar satışı SHORT potansiyeli. "
            )
        if pct <= cfg.severe_drop_pct:
            return (
                f"Son 24 saatte sert düşüş ({pct:.1f}%) devam ediyor — "
                f"momentum SHORT lehine. "
            )
        if pct <= -5:
            return f"Son 24 saatte düşüş trendi sürüyor ({pct:.1f}%). "

    if signal_kind == "BEKLE":
        if pct <= cfg.severe_drop_pct and snap.rsi < cfg.oversold_rsi_context:
            return (
                f"Coin son 24 saatte {pct:.1f}% düşmüş ve RSI {snap.rsi:.1f} ile "
                f"aşırı satım bölgesine yakın — ancak henüz net LONG/SHORT koşulu yok. "
            )
        if pct >= cfg.strong_rise_pct:
            return (
                f"Son 24 saatte güçlü yükseliş ({pct:+.1f}%) var; "
                f"giriş için ek teyit bekleniyor. "
            )

    return ""


def _build_long_reason(
    snap: IndicatorSnapshot,
    checks: list[str],
    ctx: MarketContext | None = None,
    cfg: FuturesStrategyConfig | None = None,
) -> str:
    """LONG sinyali için detaylı Türkçe gerekçe metni."""
    active_cfg = cfg or FuturesStrategyConfig()
    context_intro = _build_market_context_intro(ctx, snap, "LONG", active_cfg)
    return (
        f"{context_intro}"
        f"UZUN (LONG) pozisyon koşulları tam olarak sağlandı. "
        f"Fiyat ({snap.close:.4f}) EMA 200 ({snap.ema_200:.4f}) üzerinde; "
        f"bu durum orta-uzun vadede yükseliş trendine işaret ediyor. "
        f"EMA 20 ({snap.ema_20:.4f}) EMA 50 ({snap.ema_50:.4f}) üzerinde — "
        f"kısa vadeli momentum lehte. RSI {snap.rsi:.2f} ile güçlü ancak aşırı alım "
        f"bölgesinde değil ({active_cfg.rsi_long_min:.0f}–{active_cfg.rsi_long_max:.0f} bandı). "
        f"MACD histogramı pozitif ({snap.macd_histogram:.6f}); alıcı baskısı devam ediyor. "
        f"ATR(14) volatilite ölçümü: {snap.atr:.4f}.\n\n"
        f"Koşul özeti: {' | '.join(checks)}"
    )


def _build_early_long_reason(
    snap: IndicatorSnapshot,
    ctx: MarketContext,
    checks: list[str],
    cfg: FuturesStrategyConfig,
) -> str:
    """Riskli/Erken LONG sinyali için detaylı Türkçe gerekçe metni."""
    context_intro = _build_market_context_intro(ctx, snap, "EARLY_LONG", cfg)
    pct_text = (
        f"{ctx.price_change_pct_24h:+.1f}%"
        if ctx.price_change_pct_24h is not None
        else "—"
    )
    return (
        f"⚠️ RİSKLİ / ERKEN LONG sinyali — yüksek volatilite ve erken giriş riski taşır. "
        f"{context_intro}"
        f"Coin son 24 saatte {pct_text} değişim göstermiş; RSI {snap.rsi:.2f} ile "
        f"derin aşırı satım bölgesinde (<{cfg.early_long_rsi_max:.0f}). "
        f"Son mum yeşil (toparlanma mumu) tespit edildi. MACD henüz pozitife "
        f"dönmemiş olabilir (gecikmeli indikatör); bu nedenle standart LONG yerine "
        f"erken tepki alımı senaryosu değerlendiriliyor. "
        f"ATR(14): {snap.atr:.4f}.\n\n"
        f"Koşul özeti: {' | '.join(checks)}"
    )


def _build_short_reason(
    snap: IndicatorSnapshot,
    checks: list[str],
    ctx: MarketContext | None = None,
    cfg: FuturesStrategyConfig | None = None,
) -> str:
    """SHORT sinyali için detaylı Türkçe gerekçe metni."""
    active_cfg = cfg or FuturesStrategyConfig()
    context_intro = _build_market_context_intro(ctx, snap, "SHORT", active_cfg)
    return (
        f"{context_intro}"
        f"KISA (SHORT) pozisyon koşulları tam olarak sağlandı. "
        f"Fiyat ({snap.close:.4f}) EMA 200 ({snap.ema_200:.4f}) altında; "
        f"genel trend düşüş yönlü. EMA 20 ({snap.ema_20:.4f}) EMA 50 "
        f"({snap.ema_50:.4f}) altında — kısa vadeli satış baskısı hakim. "
        f"RSI {snap.rsi:.2f} düşüş momentumu bandında "
        f"({active_cfg.rsi_short_min:.0f}–{active_cfg.rsi_short_max:.0f}). "
        f"MACD histogramı negatif ({snap.macd_histogram:.6f}); "
        f"satıcılar kontrolü elinde tutuyor. "
        f"ATR(14) volatilite ölçümü: {snap.atr:.4f}.\n\n"
        f"Koşul özeti: {' | '.join(checks)}"
    )


def _build_wait_reason(
    snap: IndicatorSnapshot,
    long_checks: list[str],
    short_checks: list[str],
    ctx: MarketContext | None = None,
    cfg: FuturesStrategyConfig | None = None,
) -> str:
    """BEKLE sinyali için detaylı Türkçe gerekçe metni."""
    active_cfg = cfg or FuturesStrategyConfig()
    context_intro = _build_market_context_intro(ctx, snap, "BEKLE", active_cfg)
    pct_part = ""
    if ctx is not None and ctx.price_change_pct_24h is not None:
        pct_part = f"24s değişim: {ctx.price_change_pct_24h:+.2f}% | "
    return (
        f"{context_intro}"
        f"Net bir LONG veya SHORT sinyali oluşmadı — pozisyon açılmaması önerilir. "
        f"Anlık durum: Kapanış {snap.close:.4f} | EMA 20: {snap.ema_20:.4f} | "
        f"EMA 50: {snap.ema_50:.4f} | EMA 200: {snap.ema_200:.4f} | "
        f"{pct_part}"
        f"RSI(14): {snap.rsi:.2f} | MACD Histogram: {snap.macd_histogram:.6f} | "
        f"ATR(14): {snap.atr:.4f}. "
        f"LONG değerlendirmesi: {' | '.join(long_checks)}. "
        f"SHORT değerlendirmesi: {' | '.join(short_checks)}."
    )


def _evaluate_futures_signals(
    snap: IndicatorSnapshot,
    enriched: pd.DataFrame,
    cfg: FuturesStrategyConfig,
    market_context: MarketContext | None = None,
) -> dict[str, Any]:
    """İndikatör anlık görüntüsünden sinyal sözlüğü üretir."""
    long_ok, long_checks = _check_long_conditions(snap, enriched, cfg)
    short_ok, short_checks = _check_short_conditions(snap, cfg)
    early_ok, early_checks = _check_early_long_conditions(
        snap, enriched, cfg, market_context
    )

    if long_ok:
        return {
            "signal": "LONG",
            "reason": _build_long_reason(snap, long_checks, market_context, cfg),
            "atr": round(snap.atr, 6),
            "is_early_long": False,
        }

    if early_ok and market_context is not None:
        return {
            "signal": "LONG",
            "reason": _build_early_long_reason(
                snap, market_context, early_checks, cfg
            ),
            "atr": round(snap.atr, 6),
            "is_early_long": True,
        }

    if short_ok:
        return {
            "signal": "SHORT",
            "reason": _build_short_reason(snap, short_checks, market_context, cfg),
            "atr": round(snap.atr, 6),
            "is_early_long": False,
        }

    return {
        "signal": "BEKLE",
        "reason": _build_wait_reason(
            snap, long_checks, short_checks, market_context, cfg
        ),
        "atr": round(snap.atr, 6),
        "is_early_long": False,
    }


def evaluate_futures(
    df: pd.DataFrame,
    config: FuturesStrategyConfig | None = None,
    price_change_pct_24h: Optional[float] = None,
    market_context: MarketContext | None = None,
) -> dict[str, Any]:
    """
    Vadeli işlem stratejisini çalıştırır ve sözlük döndürür.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV veri çerçevesi (data_engine.fetch_ohlcv çıktısı).
    config : FuturesStrategyConfig, optional
        Strateji parametreleri.
    price_change_pct_24h : float, optional
        Binance get_ticker ``priceChangePercent`` — 24s piyasa bağlamı.
    market_context : MarketContext, optional
        ``price_change_pct_24h`` yerine doğrudan bağlam nesnesi verilebilir.

    Returns
    -------
    dict
        ``signal``  : 'LONG' | 'SHORT' | 'BEKLE'
        ``reason``  : Detaylı Türkçe gerekçe metni
        ``atr``     : ATR(14) değeri (float)
    """
    cfg = config or FuturesStrategyConfig()
    ctx = market_context
    if ctx is None and price_change_pct_24h is not None:
        ctx = MarketContext.from_ticker(price_change_pct_24h)

    enriched = add_indicators(df, cfg)
    snap = _extract_snapshot(enriched, cfg)
    return _evaluate_futures_signals(snap, enriched, cfg, ctx)


def _signal_type_from_label(label: SignalLabel) -> SignalType:
    """Sözlük sinyal etiketini SignalType'a çevirir."""
    mapping = {
        "LONG": SignalType.LONG,
        "SHORT": SignalType.SHORT,
        "BEKLE": SignalType.BEKLE,
    }
    return mapping[label]


def _result_from_dict(
    data: dict[str, Any],
    snap: IndicatorSnapshot,
    market_context: MarketContext | None = None,
) -> SignalResult:
    """Sözlük çıktısını SignalResult nesnesine dönüştürür."""
    label: SignalLabel = data["signal"]
    ctx = market_context
    return SignalResult(
        signal=_signal_type_from_label(label),
        signal_label=label,
        rsi=snap.rsi,
        macd=snap.macd,
        macd_signal=snap.macd_signal,
        macd_histogram=snap.macd_histogram,
        timestamp=snap.timestamp,
        close=snap.close,
        atr=data.get("atr"),
        ema_20=snap.ema_20,
        ema_50=snap.ema_50,
        ema_200=snap.ema_200,
        message=data.get("reason", ""),
        is_early_long=bool(data.get("is_early_long", False)),
        price_change_pct_24h=(
            ctx.price_change_pct_24h if ctx is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Strateji arayüzü
# ---------------------------------------------------------------------------


class BaseStrategy(ABC):
    """Tüm strateji sınıfları için temel arayüz."""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        market_context: MarketContext | None = None,
        price_change_pct_24h: Optional[float] = None,
    ) -> SignalResult:
        """DataFrame'i değerlendirip sinyal döndürür."""


class FuturesTrendStrategy(BaseStrategy):
    """
    Vadeli işlem trend takip stratejisi.

    LONG  : EMA 200 üstü trend + EMA 20/50 bullish + RSI 50–70 + MACD hist > 0
    SHORT : EMA 200 altı trend + EMA 20 < 50 + RSI 30–50 + MACD hist < 0
    ERKEN LONG : Sert 24s düşüş + RSI < 25 + yeşil toparlanma mumu (MACD şart değil)
    BEKLE : Koşullar sağlanmadığında
    """

    name = "futures_trend"

    def __init__(self, config: FuturesStrategyConfig | None = None) -> None:
        self.config = config or FuturesStrategyConfig()

    def evaluate(
        self,
        df: pd.DataFrame,
        market_context: MarketContext | None = None,
        price_change_pct_24h: Optional[float] = None,
    ) -> SignalResult:
        ctx = market_context
        if ctx is None and price_change_pct_24h is not None:
            ctx = MarketContext.from_ticker(price_change_pct_24h)

        enriched = add_indicators(df, self.config)
        snap = _extract_snapshot(enriched, self.config)
        data = _evaluate_futures_signals(snap, enriched, self.config, ctx)
        return _result_from_dict(data, snap, ctx)


# Geriye dönük uyumluluk
RsiMacdStrategy = FuturesTrendStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    FuturesTrendStrategy.name: FuturesTrendStrategy,
    "rsi_macd": FuturesTrendStrategy,
}


def get_strategy(name: str = "futures_trend", **kwargs) -> BaseStrategy:
    """Kayıtlı stratejilerden birini isimle oluşturur."""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY)
        raise ValueError(f"Bilinmeyen strateji: '{name}'. Mevcut: {available}")
    return STRATEGY_REGISTRY[name](**kwargs)


def _ensure_utf8_stdout() -> None:
    """Windows terminalinde emoji çıktısı için UTF-8 kodlamasını dener."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Sinyal üretimi ve ekrana yazdırma
# ---------------------------------------------------------------------------


def evaluate_signal(
    df: pd.DataFrame,
    strategy: BaseStrategy | None = None,
    price_change_pct_24h: Optional[float] = None,
    market_context: MarketContext | None = None,
) -> SignalResult:
    """
    Verilen DataFrame için strateji değerlendirmesi yapar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV verisi.
    strategy : BaseStrategy, optional
        Kullanılacak strateji.
    price_change_pct_24h : float, optional
        Binance get_ticker priceChangePercent (24s piyasa bağlamı).
    market_context : MarketContext, optional
        Doğrudan piyasa bağlamı nesnesi.

    Returns
    -------
    SignalResult
        Sinyal tipi, indikatör detayları ve gerekçe metni.
    """
    active_strategy = strategy or FuturesTrendStrategy()
    return active_strategy.evaluate(
        df,
        market_context=market_context,
        price_change_pct_24h=price_change_pct_24h,
    )


def print_signal(
    df: pd.DataFrame,
    symbol: str = "",
    interval: str = "",
    strategy: BaseStrategy | None = None,
    price_change_pct_24h: Optional[float] = None,
) -> SignalResult:
    """Stratejiyi çalıştırır ve sonucu terminale yazdırır."""
    result = evaluate_signal(
        df,
        strategy=strategy,
        price_change_pct_24h=price_change_pct_24h,
    )
    _ensure_utf8_stdout()

    header_parts = []
    if symbol:
        header_parts.append(f"Parite: {symbol.upper()}")
    if interval:
        header_parts.append(f"Zaman dilimi: {interval}")
    header = " | ".join(header_parts) if header_parts else "Sinyal Raporu"

    time_str = (
        result.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        if result.timestamp is not None
        else "—"
    )

    width = 60
    border = "=" * width
    signal_display = {
        "LONG": "🟢 LONG (UZUN)",
        "SHORT": "🔴 SHORT (KISA)",
        "BEKLE": "⚪ BEKLE",
    }.get(result.signal_label, result.signal.value)

    print()
    print(border)
    print(f"  {signal_display}".center(width))
    print(border)
    print(f"  {header}")
    print(f"  Mum zamanı : {time_str}")
    if result.close is not None:
        print(f"  Kapanış    : {result.close:.4f}")
    if result.ema_200 is not None:
        print(f"  EMA 20/50/200: {result.ema_20:.4f} / {result.ema_50:.4f} / {result.ema_200:.4f}")
    if result.rsi is not None:
        print(f"  RSI(14)    : {result.rsi:.2f}")
    if result.macd_histogram is not None:
        print(f"  MACD Hist  : {result.macd_histogram:.6f}")
    if result.atr is not None:
        print(f"  ATR(14)    : {result.atr:.4f}")
    if result.price_change_pct_24h is not None:
        print(f"  24s Değişim: %{result.price_change_pct_24h:+.2f}")
    if result.is_early_long:
        print("  Not        : ⚠️ Riskli/Erken LONG")
    print(f"  Gerekçe    : {result.message}")
    print(border)
    print()

    return result
