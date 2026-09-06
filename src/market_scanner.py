"""
Piyasa tarayıcı modülü.

Binance USDT paritelerinden akıllı coin havuzu oluşturur (trend, düşüş,
hacim patlaması), LONG ve SHORT sinyallerini bulur ve risk seviyeleriyle
birlikte tavsiye özeti üretir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import time
from typing import Any, Callable, Literal, Optional

import pandas as pd

from .data_engine import fetch_all_tickers_24hr, fetch_ohlcv
from .risk_manager import (
    ExitLevels,
    RiskConfig,
    calculate_exit_levels_from_signal,
    estimate_target_eta,
)
from .strategy_engine import BaseStrategy, FuturesTrendStrategy, SignalResult, SignalType


# Kara liste — stablecoin / fiat baz varlıklar (USDT paritesi olsa bile atlanır)
BLACKLIST_BASES: frozenset[str] = frozenset(
    {
        "USDC",
        "FDUSD",
        "TUSD",
        "BUSD",
        "EUR",
        "USD1",
        "USDP",
        "DAI",
        "GBP",
    }
)

# Kaldıraçlı / ters ETF benzeri pariteler
_EXCLUDED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")

# Her parite taraması sonrası bekleme süresi (saniye) — Binance rate limit koruması
_TARAMA_BEKLEME_SANIYE = 0.1

# Akıllı havuz hedefi — örtüşme olursa hacimden tamamlanır
_MIN_HAVUZ_BOYUTU = 60

PoolCategory = Literal["trend", "loser", "volume_surge", "dinamik"]


@dataclass(frozen=True)
class ScannerConfig:
    """Piyasa tarayıcı yapılandırması."""

    interval: str = "1h"
    history_limit: int = 500
    request_delay_sec: float = _TARAMA_BEKLEME_SANIYE
    quote_asset: str = "USDT"
    min_quote_volume: float = 0.0
    pool_category_size: int = 20
    volume_surge_max_change_pct: float = 8.0
    use_smart_pool: bool = True
    top_n: int = 50


@dataclass
class ScanStats:
    """Tarama istatistikleri."""

    incelenen: int = 0
    atlanan: int = 0
    long_sinyali: int = 0
    short_sinyali: int = 0
    toplam: int = 0
    trend_sayisi: int = 0
    loser_sayisi: int = 0
    hacim_patlamasi_sayisi: int = 0
    dinamik_sayisi: int = 0
    mesaj: str = ""

    @property
    def toplam_sinyal(self) -> int:
        return self.long_sinyali + self.short_sinyali


@dataclass
class ScanOpportunity:
    """LONG veya SHORT sinyali veren tek bir coin fırsatı."""

    symbol: str
    signal: SignalType
    signal_label: str
    entry_price: float
    current_price: float
    rsi: Optional[float]
    atr: Optional[float]
    volume_24h_usdt: float
    price_change_pct_24h: float
    stop_loss: Optional[float]
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    reason: str
    recommendation: str
    pool_sources: list[str] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_signal: Optional[SignalResult] = None
    exit_levels: Optional[ExitLevels] = None
    interval: str = "1h"
    tp1_eta_text: Optional[str] = None
    tp2_eta_text: Optional[str] = None


def _log_bilgi(mesaj: str) -> None:
    """Bilgilendirme mesajını terminale yazar."""
    print(f"[Bilgi] {mesaj}")


def _log_uyari(mesaj: str) -> None:
    """Uyarı mesajını terminale yazar."""
    print(f"[Uyarı] {mesaj}")


def _log_hata(mesaj: str) -> None:
    """Hata mesajını terminale yazar."""
    print(f"[Hata] {mesaj}")


def _extract_base(symbol: str, quote_asset: str = "USDT") -> str:
    """Parite sembolünden baz varlık kodunu çıkarır."""
    if symbol.endswith(quote_asset):
        return symbol[: -len(quote_asset)]
    return symbol


def _is_blacklisted(symbol: str, quote_asset: str = "USDT") -> bool:
    """Baz varlık kara listede mi kontrol eder (tam eşleşme veya önek)."""
    base = _extract_base(symbol, quote_asset)
    for blocked in sorted(BLACKLIST_BASES, key=len, reverse=True):
        if base == blocked or base.startswith(blocked):
            return True
    return False


def is_blacklisted_symbol(symbol: str, quote_asset: str = "USDT") -> bool:
    """Stablecoin / fiat bazlı paritenin kara listede olup olmadığını döndürür."""
    return _is_blacklisted(symbol, quote_asset)


def _is_valid_usdt_symbol(symbol: str, quote_asset: str = "USDT") -> bool:
    """USDT paritesinin taranmaya uygun olup olmadığını kontrol eder."""
    if not symbol.endswith(quote_asset):
        return False

    for suffix in _EXCLUDED_SUFFIXES:
        if symbol.endswith(suffix):
            return False

    if _is_blacklisted(symbol, quote_asset):
        return False

    base = _extract_base(symbol, quote_asset)
    if len(base) < 2:
        return False

    return True


def _parse_ticker_row(
    ticker: dict[str, Any],
    quote_asset: str = "USDT",
    min_quote_volume: float = 0.0,
) -> Optional[dict[str, Any]]:
    """Ham ticker kaydını taranabilir aday sözlüğüne dönüştürür."""
    symbol = ticker.get("symbol", "")
    if not _is_valid_usdt_symbol(symbol, quote_asset):
        return None

    try:
        quote_volume = float(ticker.get("quoteVolume", 0))
        if quote_volume < min_quote_volume:
            return None

        trade_count = float(ticker.get("count", 0) or 0)
        price_change_pct = float(ticker.get("priceChangePercent", 0))
        return {
            "symbol": symbol,
            "quote_volume": quote_volume,
            "price_change_pct": price_change_pct,
            "last_price": float(ticker.get("lastPrice", 0)),
            "trade_count": trade_count,
            "activity_score": _activity_score(price_change_pct, quote_volume, trade_count),
        }
    except (TypeError, ValueError):
        return None


def _activity_score(
    price_change_pct: float,
    quote_volume: float,
    trade_count: float = 0.0,
) -> float:
    """24s fiyat değişimi, hacim ve işlem sayısıyla anlık hareket skoru."""
    magnitude = abs(float(price_change_pct) or 0.0)
    volume = max(float(quote_volume) or 0.0, 0.0)
    trades = max(float(trade_count) or 0.0, 0.0)
    return magnitude * math.log10(1.0 + volume) * (1.0 + 0.15 * math.log10(1.0 + trades))


def _select_dynamic_movers(
    candidates: list[dict[str, Any]],
    category_size: int,
) -> list[dict[str, Any]]:
    """
    O an piyasada hareketli coinleri seçer — yalnızca popüler/hacimli olanlar değil.

    Skor: |24s değişim| × log(hacim) × işlem sayısı katkısı.
    Hacim tabanı medyanın %5'i; değişim eşiği piyasaya göre dinamik.
    """
    if not candidates:
        return []

    volumes = [float(item["quote_volume"]) for item in candidates]
    changes = [abs(float(item["price_change_pct"])) for item in candidates]
    volume_floor = max(float(pd.Series(volumes).median()) * 0.05, 50_000.0)
    change_floor = max(3.0, float(pd.Series(changes).quantile(0.55)))

    hot_count = sum(1 for item in candidates if abs(item["price_change_pct"]) >= 8.0)
    dinamik_size = min(category_size + min(hot_count // 8, 15), 40)

    hareketliler = [
        item
        for item in candidates
        if item["quote_volume"] >= volume_floor
        and abs(item["price_change_pct"]) >= change_floor
    ]
    if len(hareketliler) < dinamik_size:
        hareketliler = [
            item for item in candidates if item["quote_volume"] >= volume_floor
        ]

    hareketliler.sort(
        key=lambda item: float(item.get("activity_score") or 0.0),
        reverse=True,
    )
    return hareketliler[:dinamik_size]


def fetch_usdt_ticker_candidates(
    quote_asset: str = "USDT",
    min_quote_volume: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Binance 24s ticker verisinden taranabilir USDT paritelerini döndürür.

    Kara listedeki stablecoin/fiat bazlı pariteler filtrelenir.
    """
    try:
        tickers = fetch_all_tickers_24hr()
    except Exception:
        _log_hata(
            "Binance halka açık /api/v3/ticker/24hr uç noktasından veri alınamadı. "
            "Havuz taraması boş sonuçla sonlanacak, uygulama çalışmaya devam ediyor."
        )
        return []

    if not tickers:
        _log_uyari("Binance 24s ticker yanıtı boş döndü. Taranacak parite yok.")
        return []

    candidates: list[dict[str, Any]] = []
    for ticker in tickers:
        row = _parse_ticker_row(
            ticker,
            quote_asset=quote_asset,
            min_quote_volume=min_quote_volume,
        )
        if row is not None:
            candidates.append(row)

    _log_bilgi(
        f"24s ticker verisinden {len(candidates)} taranabilir USDT paritesi elde edildi "
        f"(kara liste ve hacim filtresi uygulandı)."
    )
    return candidates


def build_smart_coin_pool(
    category_size: int = 20,
    quote_asset: str = "USDT",
    min_quote_volume: float = 0.0,
    volume_surge_max_change_pct: float = 8.0,
    min_pool_size: int = _MIN_HAVUZ_BOYUTU,
) -> list[dict[str, Any]]:
    """
    Akıllı coin havuzu oluşturur — 24s hacim ve fiyat değişimine göre
    o an hareketli coinleri dinamik olarak dahil eder (hedef 60+).

    Kategoriler:
    - trend: En çok yükselenler
    - loser: En çok düşenler
    - volume_surge: Yüksek hacim, fiyatı henüz aşırı şişmemiş pariteler
    - dinamik: |değişim| × log(hacim) skoruyla anlık hareketliler
    """
    candidates = fetch_usdt_ticker_candidates(
        quote_asset=quote_asset,
        min_quote_volume=min_quote_volume,
    )

    if not candidates:
        _log_uyari("Akıllı havuz için aday parite bulunamadı.")
        return []

    # 1) Trend olanlar — en yüksek priceChangePercent
    trend_olanlar = sorted(
        candidates,
        key=lambda item: item["price_change_pct"],
        reverse=True,
    )[:category_size]

    # 2) Düşen bıçaklar — en düşük priceChangePercent
    dusen_bicaklar = sorted(
        candidates,
        key=lambda item: item["price_change_pct"],
    )[:category_size]

    # 3) Hacim patlaması — yüksek hacim, fiyat henüz çok şişmemiş
    hacim_adaylari = [
        item
        for item in candidates
        if abs(item["price_change_pct"]) <= volume_surge_max_change_pct
    ]
    hacim_adaylari.sort(key=lambda item: item["quote_volume"], reverse=True)
    hacim_patlamasi = hacim_adaylari[:category_size]

    # Yeterli aday yoksa: yüksek hacimli ama en az fiyat hareketi olanlardan tamamla
    if len(hacim_patlamasi) < category_size:
        mevcut = {item["symbol"] for item in hacim_patlamasi}
        yedek = sorted(
            candidates,
            key=lambda item: (abs(item["price_change_pct"]), -item["quote_volume"]),
        )
        for item in yedek:
            if item["symbol"] in mevcut:
                continue
            hacim_patlamasi.append(item)
            mevcut.add(item["symbol"])
            if len(hacim_patlamasi) >= category_size:
                break

    # 4) 24s dinamik hareketliler — popüler olmayan ama o an hareket edenler
    dinamik_olanlar = _select_dynamic_movers(candidates, category_size)

    birlesik: list[dict[str, Any]] = []
    gorulen: set[str] = set()

    for kaynak, grup in (
        ("dinamik", dinamik_olanlar),
        ("trend", trend_olanlar),
        ("loser", dusen_bicaklar),
        ("volume_surge", hacim_patlamasi),
    ):
        for item in grup:
            symbol = item["symbol"]
            if symbol in gorulen:
                continue
            gorulen.add(symbol)
            birlesik.append({**item, "pool_sources": [kaynak]})

    # Aynı coin birden fazla kategorideyse etiketleri birleştir
    for kaynak, grup in (
        ("dinamik", dinamik_olanlar),
        ("trend", trend_olanlar),
        ("loser", dusen_bicaklar),
        ("volume_surge", hacim_patlamasi),
    ):
        grup_sembolleri = {item["symbol"] for item in grup}
        for entry in birlesik:
            if entry["symbol"] in grup_sembolleri and kaynak not in entry["pool_sources"]:
                entry["pool_sources"].append(kaynak)

    hedef = max(min_pool_size, category_size)
    if len(birlesik) < hedef:
        skor_sirali = sorted(
            candidates,
            key=lambda item: float(item.get("activity_score") or 0.0),
            reverse=True,
        )
        for item in skor_sirali:
            if item["symbol"] in gorulen:
                continue
            gorulen.add(item["symbol"])
            birlesik.append({**item, "pool_sources": ["dinamik"]})
            if len(birlesik) >= hedef:
                break
        _log_bilgi(
            f"Havuz örtüşme nedeniyle {hedef} altına düştü; "
            f"24s hareket skorundan tamamlandı — yeni toplam: {len(birlesik)}."
        )

    _log_bilgi(
        f"Akıllı havuz oluşturuldu — dinamik: {len(dinamik_olanlar)}, "
        f"trend: {len(trend_olanlar)}, düşen: {len(dusen_bicaklar)}, "
        f"hacim patlaması: {len(hacim_patlamasi)}, "
        f"benzersiz toplam: {len(birlesik)} (hedef: {hedef}+)."
    )
    return birlesik


def get_top_volume_symbols(
    top_n: int = 50,
    quote_asset: str = "USDT",
    min_quote_volume: float = 0.0,
) -> list[dict[str, Any]]:
    """
    24 saatlik hacme göre en yüksek USDT paritelerini döndürür.

    Geriye dönük uyumluluk için korunmuştur; yeni taramalar ``build_smart_coin_pool``
    kullanır.
    """
    candidates = fetch_usdt_ticker_candidates(
        quote_asset=quote_asset,
        min_quote_volume=min_quote_volume,
    )
    candidates.sort(key=lambda item: item["quote_volume"], reverse=True)
    secilen = candidates[:top_n]
    _log_bilgi(
        f"Hacme göre ilk {len(secilen)} USDT paritesi seçildi "
        f"(toplam aday: {len(candidates)})."
    )
    return secilen


def _is_active_signal(signal_result: SignalResult) -> bool:
    """LONG veya SHORT sinyali olup olmadığını kontrol eder."""
    if signal_result.signal_label in ("LONG", "SHORT"):
        return True
    return signal_result.signal in (
        SignalType.LONG,
        SignalType.BUY,
        SignalType.SHORT,
        SignalType.SELL,
    )


def _resolve_signal_label(signal_result: SignalResult) -> str:
    """Sinyal etiketini LONG veya SHORT olarak döndürür."""
    if signal_result.signal_label in ("LONG", "SHORT"):
        return signal_result.signal_label
    if signal_result.signal in (SignalType.LONG, SignalType.BUY):
        return "LONG"
    if signal_result.signal in (SignalType.SHORT, SignalType.SELL):
        return "SHORT"
    return "BEKLE"


def build_risk_summary(exit_levels: Optional[ExitLevels]) -> str:
    """Risk seviyelerinden kısa Türkçe özet üretir."""
    if exit_levels is None:
        return "Risk seviyeleri hesaplanamadı."
    return (
        f"Zarar durdur: {exit_levels.stop_loss:.4f} | "
        f"Kar hedefi 1: {exit_levels.take_profit_1:.4f} | "
        f"Kar hedefi 2: {exit_levels.take_profit_2:.4f} | "
        f"ATR(14): {exit_levels.atr:.4f}"
    )


# Geriye dönük uyumluluk (src.__init__ export'u)
build_recommendation_summary = build_risk_summary


def _analyze_symbol(
    symbol: str,
    ticker_meta: dict[str, Any],
    interval: str,
    history_limit: int,
    strategy: BaseStrategy,
    risk_config: RiskConfig,
) -> Optional[ScanOpportunity]:
    """
    Tek parite için mum verisi çeker, strateji ve risk analizi yapar.

    Yalnızca LONG veya SHORT sinyali varsa ScanOpportunity döndürür.
    Kara listedeki stablecoin pariteleri atlanır.
    """
    if is_blacklisted_symbol(symbol):
        return None

    # Adım 1: Mum verisi — boş veya hatalı yanıtta bu coin atlanır
    try:
        df = fetch_ohlcv(symbol=symbol, interval=interval, limit=history_limit)
    except Exception as veri_hatasi:
        raise RuntimeError(
            f"{symbol} için mum verisi alınamadı"
        ) from veri_hatasi

    if df is None or getattr(df, "empty", True):
        raise RuntimeError(f"{symbol} için mum verisi boş döndü")

    # Adım 2: Strateji / indikatör analizi
    try:
        signal_result = strategy.evaluate(
            df,
            price_change_pct_24h=ticker_meta.get("price_change_pct"),
        )
    except Exception as analiz_hatasi:
        raise RuntimeError(
            f"{symbol} için indikatör veya sinyal hesaplanamadı"
        ) from analiz_hatasi

    if not _is_active_signal(signal_result):
        return None

    signal_label = _resolve_signal_label(signal_result)

    # Adım 3: Risk seviyeleri (başarısız olursa kart yine de üretilir)
    exit_levels: Optional[ExitLevels] = None
    try:
        exit_levels = calculate_exit_levels_from_signal(signal_result, config=risk_config)
    except Exception:
        _log_uyari(f"{symbol} için zarar durdur / kar hedefi seviyeleri hesaplanamadı.")

    reason = signal_result.message or "Strateji gerekçesi üretilemedi."
    risk_summary = build_risk_summary(exit_levels)
    recommendation = (
        f"{reason} | {risk_summary} | "
        f"24s hacim: {ticker_meta['quote_volume']:,.0f} USDT, "
        f"24s değişim: %{ticker_meta['price_change_pct']:+.2f}."
    )

    entry = float(signal_result.close or ticker_meta["last_price"])
    atr_value = float(signal_result.atr) if signal_result.atr else None
    tp1 = exit_levels.take_profit_1 if exit_levels else None
    tp2 = exit_levels.take_profit_2 if exit_levels else None
    tp1_eta = (
        estimate_target_eta(entry, tp1, atr_value, interval)
        if tp1 is not None and atr_value
        else None
    )
    tp2_eta = (
        estimate_target_eta(entry, tp2, atr_value, interval)
        if tp2 is not None and atr_value
        else None
    )

    return ScanOpportunity(
        symbol=symbol,
        signal=signal_result.signal,
        signal_label=signal_label,
        entry_price=entry,
        current_price=float(ticker_meta["last_price"]),
        rsi=signal_result.rsi,
        atr=signal_result.atr,
        volume_24h_usdt=ticker_meta["quote_volume"],
        price_change_pct_24h=ticker_meta["price_change_pct"],
        stop_loss=exit_levels.stop_loss if exit_levels else None,
        take_profit_1=tp1,
        take_profit_2=tp2,
        reason=reason,
        recommendation=recommendation,
        pool_sources=list(ticker_meta.get("pool_sources") or []),
        raw_signal=signal_result,
        exit_levels=exit_levels,
        interval=interval,
        tp1_eta_text=tp1_eta.text if tp1_eta else None,
        tp2_eta_text=tp2_eta.text if tp2_eta else None,
    )


def scan_market(
    config: ScannerConfig | None = None,
    strategy: BaseStrategy | None = None,
    risk_config: RiskConfig | None = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[list[ScanOpportunity], ScanStats]:
    """
    Binance USDT paritelerini tarar ve LONG / SHORT fırsatlarını döndürür.

    Her parite ayrı try-except ile korunur; hata olsa bile tarama sürer.
    """
    cfg = config or ScannerConfig()
    active_strategy = strategy or FuturesTrendStrategy()
    active_risk = risk_config or RiskConfig()
    bekleme = (
        cfg.request_delay_sec
        if cfg.request_delay_sec and cfg.request_delay_sec > 0
        else _TARAMA_BEKLEME_SANIYE
    )

    if cfg.use_smart_pool:
        _log_bilgi(
            f"Akıllı havuz taraması başlıyor — kategori başına {cfg.pool_category_size} parite "
            f"(trend + düşen + hacim patlaması), zaman dilimi: {cfg.interval}."
        )
        pool_builder = lambda: build_smart_coin_pool(
            category_size=cfg.pool_category_size,
            quote_asset=cfg.quote_asset,
            min_quote_volume=cfg.min_quote_volume,
            volume_surge_max_change_pct=cfg.volume_surge_max_change_pct,
            min_pool_size=_MIN_HAVUZ_BOYUTU,
        )
    else:
        _log_bilgi(
            f"Piyasa taraması başlıyor — hacme göre ilk {cfg.top_n} USDT paritesi, "
            f"zaman dilimi: {cfg.interval}."
        )
        pool_builder = lambda: get_top_volume_symbols(
            top_n=cfg.top_n,
            quote_asset=cfg.quote_asset,
            min_quote_volume=cfg.min_quote_volume,
        )

    try:
        scan_pool = pool_builder()
    except Exception:
        mesaj = (
            "Parite havuzu oluşturulamadı. Binance bağlantısını kontrol edin; "
            "tarama durduruldu, uygulama çalışmaya devam ediyor."
        )
        _log_hata(mesaj)
        return [], ScanStats(mesaj=mesaj)

    if not scan_pool:
        mesaj = (
            "Taranacak parite bulunamadı. 24s ticker boş döndü veya "
            "filtreler (kara liste / hacim) tüm adayları eledi."
        )
        _log_uyari(mesaj)
        return [], ScanStats(mesaj=mesaj)

    pool_trend = sum(1 for item in scan_pool if "trend" in item.get("pool_sources", []))
    pool_loser = sum(1 for item in scan_pool if "loser" in item.get("pool_sources", []))
    pool_surge = sum(
        1 for item in scan_pool if "volume_surge" in item.get("pool_sources", [])
    )
    pool_dinamik = sum(
        1 for item in scan_pool if "dinamik" in item.get("pool_sources", [])
    )

    opportunities: list[ScanOpportunity] = []
    total = len(scan_pool)
    atlanan_sayisi = 0
    incelenen_sayisi = 0
    long_sayisi = 0
    short_sayisi = 0

    _log_bilgi(
        f"Toplam {total} benzersiz parite akıllı havuzdan analiz edilecek "
        f"(dinamik: {pool_dinamik}, trend: {pool_trend}, düşen: {pool_loser}, "
        f"hacim: {pool_surge})."
    )

    for index, ticker_meta in enumerate(scan_pool):
        symbol = ticker_meta["symbol"]
        sira = index + 1

        if progress_callback is not None:
            try:
                progress_callback(sira, total, symbol)
            except Exception:
                pass

        _log_bilgi(f"[{sira}/{total}] {symbol} inceleniyor...")

        try:
            opportunity = _analyze_symbol(
                symbol=symbol,
                ticker_meta=ticker_meta,
                interval=cfg.interval,
                history_limit=cfg.history_limit,
                strategy=active_strategy,
                risk_config=active_risk,
            )

            incelenen_sayisi += 1

            if opportunity is None:
                _log_bilgi(f"{symbol} — LONG/SHORT sinyali yok, sonraki pariteye geçiliyor.")
            elif is_blacklisted_symbol(opportunity.symbol):
                atlanan_sayisi += 1
                _log_uyari(f"{symbol} kara listede — arayüze yansıtılmıyor, atlanıyor.")
            else:
                opportunities.append(opportunity)
                if opportunity.signal_label == "LONG":
                    long_sayisi += 1
                else:
                    short_sayisi += 1
                rsi_goster = (
                    f"{opportunity.rsi:.1f}" if opportunity.rsi is not None else "—"
                )
                _log_bilgi(
                    f"{symbol} — {opportunity.signal_label} sinyali bulundu! "
                    f"RSI: {rsi_goster}"
                )

        except Exception as tarama_hatasi:
            atlanan_sayisi += 1
            _log_uyari(
                f"{symbol} atlandı ({tarama_hatasi}). "
                f"Tarama diğer paritelerle devam ediyor."
            )
            continue

        finally:
            # Binance rate limit için istekler arası kısa bekleme
            time.sleep(bekleme)

    opportunities.sort(key=lambda item: item.volume_24h_usdt, reverse=True)

    _log_bilgi(
        f"Tarama tamamlandı. İncelenen: {incelenen_sayisi}, "
        f"atlanan: {atlanan_sayisi}, LONG: {long_sayisi}, SHORT: {short_sayisi}."
    )

    if not opportunities:
        if incelenen_sayisi == 0 and atlanan_sayisi > 0:
            sonuc_mesaji = (
                f"Tarama tamamlandı ancak {atlanan_sayisi} paritenin hiçbiri "
                f"için geçerli mum verisi alınamadı. LONG/SHORT sonucu yok."
            )
        elif atlanan_sayisi > 0:
            sonuc_mesaji = (
                f"Şu an LONG veya SHORT sinyali bulunamadı "
                f"(incelenen: {incelenen_sayisi}, atlanan: {atlanan_sayisi}). "
                f"Pozisyon açılmaması önerilir."
            )
        else:
            sonuc_mesaji = (
                f"Şu an LONG veya SHORT sinyali bulunamadı "
                f"(incelenen: {incelenen_sayisi}). Pozisyon açılmaması önerilir."
            )
        _log_uyari(sonuc_mesaji)
    else:
        sonuc_mesaji = (
            f"Tarama tamamlandı. İncelenen: {incelenen_sayisi}, "
            f"atlanan: {atlanan_sayisi}, LONG: {long_sayisi}, SHORT: {short_sayisi}."
        )

    stats = ScanStats(
        incelenen=incelenen_sayisi,
        atlanan=atlanan_sayisi,
        long_sinyali=long_sayisi,
        short_sinyali=short_sayisi,
        toplam=total,
        trend_sayisi=pool_trend,
        loser_sayisi=pool_loser,
        hacim_patlamasi_sayisi=pool_surge,
        dinamik_sayisi=pool_dinamik,
        mesaj=sonuc_mesaji,
    )
    return opportunities, stats


def _sinyal_etiketi(signal: SignalType) -> str:
    """Sinyal tipini Türkçe etikete çevirir."""
    esleme = {
        SignalType.LONG: "LONG",
        SignalType.BUY: "LONG",
        SignalType.SHORT: "SHORT",
        SignalType.SELL: "SHORT",
        SignalType.BEKLE: "BEKLE",
        SignalType.HOLD: "BEKLE",
    }
    return esleme.get(signal, "BEKLE")


def opportunities_to_dataframe(
    opportunities: list[ScanOpportunity],
) -> pd.DataFrame:
    """
    ScanOpportunity listesini kolay işlenebilir DataFrame'e dönüştürür.
    """
    if not opportunities:
        return pd.DataFrame(
            columns=[
                "parite",
                "sinyal",
                "guncel_fiyat",
                "tavsiye_giris",
                "rsi",
                "atr",
                "hacim_24s_usdt",
                "degisim_24s_yuzde",
                "zarar_durdur",
                "kar_hedefi_1",
                "kar_hedefi_2",
                "kh1_tahmini_sure",
                "kh2_tahmini_sure",
                "gerekce",
                "havuz_kaynaklari",
                "tavsiye_ozeti",
                "taranma_zamani",
            ]
        )

    rows = [
        {
            "parite": opp.symbol,
            "sinyal": opp.signal_label,
            "guncel_fiyat": opp.current_price,
            "tavsiye_giris": opp.entry_price,
            "rsi": opp.rsi,
            "atr": opp.atr,
            "hacim_24s_usdt": opp.volume_24h_usdt,
            "degisim_24s_yuzde": opp.price_change_pct_24h,
            "zarar_durdur": opp.stop_loss,
            "kar_hedefi_1": opp.take_profit_1,
            "kar_hedefi_2": opp.take_profit_2,
            "kh1_tahmini_sure": opp.tp1_eta_text,
            "kh2_tahmini_sure": opp.tp2_eta_text,
            "gerekce": opp.reason,
            "havuz_kaynaklari": ", ".join(opp.pool_sources) if opp.pool_sources else "—",
            "tavsiye_ozeti": opp.recommendation,
            "taranma_zamani": opp.scanned_at,
        }
        for opp in opportunities
    ]

    return pd.DataFrame(rows)


def scan_market_dataframe(
    config: ScannerConfig | None = None,
    strategy: BaseStrategy | None = None,
    risk_config: RiskConfig | None = None,
) -> pd.DataFrame:
    """
    Piyasayı tarar ve sonuçları DataFrame olarak döndürür.
    """
    opportunities, _ = scan_market(
        config=config,
        strategy=strategy,
        risk_config=risk_config,
    )
    return opportunities_to_dataframe(opportunities)
