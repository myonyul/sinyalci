"""
Veri motoru modülü.

Binance halka açık REST API üzerinden (API key gerekmez) geçmiş OHLCV
(Open, High, Low, Close, Volume) mum verilerini çeker, temizler ve analiz
için hazır bir Pandas DataFrame döndürür. python-binance Client kullanılmaz.
"""

from __future__ import annotations

from typing import Any, Optional
import json

import pandas as pd
import requests
from requests.exceptions import RequestException


# requests oturumunun süresiz asılı kalmaması için varsayılan zaman aşımı (saniye)
REQUEST_TIMEOUT = 15

# Halka açık REST uçları — API key gerekmez; python-binance Client kullanılmaz
_BINANCE_REST_HOSTS = (
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://data-api.binance.vision",
)
_KLINES_PATH = "/api/v3/klines"
_TICKER_PATH = "/api/v3/ticker/24hr"
_PRICE_PATH = "/api/v3/ticker/price"
_KLINES_LIMIT_MAX = 1000

_session: Optional[requests.Session] = None

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _get_session() -> requests.Session:
    """Tekil requests oturumu (User-Agent + bağlantı yeniden kullanımı)."""
    global _session
    if _session is None:
        session = requests.Session()
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            }
        )
        _session = session
    return _session


def _empty_ohlcv() -> pd.DataFrame:
    """Analiz çağrılarının sütun beklediği boş OHLCV çerçevesi."""
    return pd.DataFrame(columns=_OHLCV_COLUMNS)


def _uyari(mesaj: str) -> str:
    """Uyarıyı terminale yazar ve aynı metni döndürür."""
    print(f"[Uyarı] {mesaj}")
    return mesaj


def _to_millis(value: Optional[str | int | float]) -> Optional[int]:
    """Tarih metnini veya sayısal değeri milisaniye zaman damgasına çevirir."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Geçersiz zaman değeri.")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    parsed = pd.to_datetime(text, utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Zaman değeri çözümlenemedi: {value}")
    return int(parsed.timestamp() * 1000)


def _raise_for_binance(response: requests.Response) -> Any:
    """HTTP ve Binance JSON hata gövdesini anlamlı istisnaya çevirir."""
    try:
        payload = response.json()
    except ValueError as exc:
        response.raise_for_status()
        raise ValueError(f"Binance yanıtı JSON değil: {response.text[:200]}") from exc

    if isinstance(payload, dict) and "code" in payload and payload.get("code") not in (0, 200):
        raise ValueError(
            f"Binance API hatası ({payload.get('code')}): {payload.get('msg', payload)}"
        )
    if response.status_code >= 400:
        response.raise_for_status()
    return payload


def _public_get(path: str, params: dict[str, Any]) -> Any:
    """Halka açık Binance REST endpoint'ine GET atar; host yedekleri dener."""
    last_error: Exception | None = None
    session = _get_session()
    for host in _BINANCE_REST_HOSTS:
        url = f"{host}{path}"
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            return _raise_for_binance(response)
        except (RequestException, TimeoutError, OSError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Binance REST isteği gönderilemedi.")


def _fetch_klines_page(
    symbol: str,
    interval: str,
    limit: int,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> list:
    """Tek sayfa mum verisini public /api/v3/klines üzerinden çeker."""
    params: dict[str, Any] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    payload = _public_get(_KLINES_PATH, params)
    if not isinstance(payload, list):
        raise ValueError("Binance klines yanıtı liste değil.")
    return payload


def _fetch_raw_klines(
    symbol: str,
    interval: str,
    limit: int,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> list:
    """Gerekirse sayfalayarak ham mum listesini döndürür."""
    start_ms = _to_millis(start_time)
    end_ms = _to_millis(end_time)

    if start_ms is None:
        return _fetch_klines_page(
            symbol,
            interval,
            max(1, min(limit, _KLINES_LIMIT_MAX)),
            end_ms=end_ms,
        )

    raw_klines: list = []
    cursor = start_ms
    while True:
        page = _fetch_klines_page(
            symbol,
            interval,
            _KLINES_LIMIT_MAX,
            start_ms=cursor,
            end_ms=end_ms,
        )
        if not page:
            break
        raw_klines.extend(page)
        last_open = int(page[-1][0])
        next_cursor = last_open + 1
        if len(page) < _KLINES_LIMIT_MAX:
            break
        if end_ms is not None and next_cursor > end_ms:
            break
        cursor = next_cursor
    return raw_klines


def fetch_ohlcv(
    symbol: str,
    interval: str,
    limit: int = 500,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Belirtilen parite ve zaman dilimi için geçmiş OHLCV verisini çeker.

    Parameters
    ----------
    symbol : str
        İşlem çifti (örn: 'BTCUSDT', 'ETHUSDT').
    interval : str
        Mum zaman dilimi (örn: '1h', '15m', '4h', '1d').
    limit : int, optional
        Çekilecek mum sayısı. Varsayılan 500, maksimum 1000.
        ``start_time`` verildiğinde bu parametre yok sayılır.
    start_time : str, optional
        Başlangıç zamanı (örn: '1 Jan, 2024', '30 days ago UTC').
    end_time : str, optional
        Bitiş zamanı. Yalnızca ``start_time`` ile birlikte kullanılır.

    Returns
    -------
    pd.DataFrame
        Zaman damgası index'li, yalnızca Open, High, Low, Close, Volume
        sütunlarını içeren temizlenmiş veri çerçevesi.
        Ağ / API hatası veya boş yanıtta boş DataFrame döner (çökmez).

    Raises
    ------
    ValueError
        Geçersiz parametre verildiğinde.
    """
    # Parite sembolünü Binance formatına uygun hale getir (büyük harf)
    symbol = symbol.upper().strip()
    interval = interval.strip()

    if not symbol:
        raise ValueError("Parite (symbol) boş olamaz.")
    if not interval:
        raise ValueError("Zaman dilimi (interval) boş olamaz.")

    try:
        raw_klines = _fetch_raw_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )
    except (
        RequestException,
        TimeoutError,
        ValueError,
        OSError,
    ) as exc:
        _uyari(
            f"'{symbol}' paritesi için '{interval}' mum verisi alınamadı "
            f"(zaman aşımı veya bağlantı hatası olabilir): {exc}"
        )
        return _empty_ohlcv()
    except Exception as exc:
        _uyari(
            f"'{symbol}' paritesi için beklenmeyen mum verisi hatası: {exc}"
        )
        return _empty_ohlcv()

    if not raw_klines:
        _uyari(
            f"'{symbol}' paritesi için '{interval}' diliminde mum verisi boş döndü. "
            f"Sinyal üretimi atlandı."
        )
        return _empty_ohlcv()

    df = _clean_klines(raw_klines)
    if df.empty:
        _uyari(
            f"'{symbol}' paritesi için '{interval}' diliminde geçerli OHLCV satırı yok. "
            f"Sinyal üretimi atlandı."
        )
        return _empty_ohlcv()

    return df


def _clean_klines(raw_klines: list) -> pd.DataFrame:
    """
    Ham Binance kline listesini temiz bir OHLCV DataFrame'e dönüştürür.

    Parameters
    ----------
    raw_klines : list
        Binance API'den dönen ham mum verisi.

    Returns
    -------
    pd.DataFrame
        Temizlenmiş OHLCV veri çerçevesi.
    """
    # Ham veriyi sütun isimleriyle DataFrame'e aktar
    df = pd.DataFrame(
        raw_klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    # Milisaniye cinsinden açılış zamanını okunabilir datetime'a çevir
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    # Fiyat ve hacim sütunlarını sayısal (float) tipe dönüştür
    ohlcv_columns = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eksik (NaN) değer içeren satırları kaldır — bozuk veya eksik mumları temizler
    df = df.dropna(subset=ohlcv_columns)

    # Zaman sütununu index yap
    df = df.set_index("open_time")
    df.index.name = None

    # Kronolojik sıraya göre diz
    df = df.sort_index()

    # Aynı zaman damgasına sahip tekrarlayan kayıtları sil, son kaydı tut
    df = df[~df.index.duplicated(keep="last")]

    # Analiz için yalnızca OHLCV sütunlarını seç ve standart isimlendir
    df = df[ohlcv_columns]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    return df


def fetch_all_tickers_24hr() -> list[dict[str, Any]]:
    """
    Tüm spot paritelerin 24 saatlik istatistiklerini çeker.

    ``GET https://api.binance.com/api/v3/ticker/24hr`` — API key gerekmez.
    """
    payload = _public_get(_TICKER_PATH, {})
    if not isinstance(payload, list):
        raise ValueError("Binance 24s ticker yanıtı liste değil.")
    return payload


def fetch_live_tickers(symbols: list[str]) -> dict[str, dict[str, float]]:
    """
    Seçili paritelerin anlık fiyat ve 24s değişimini tek istekle çeker.

    ``GET /api/v3/ticker/24hr?symbols=[...]`` — başarısızsa
    ``/api/v3/ticker/price`` yedeğine düşer.
    """
    unique = sorted({str(symbol).upper().strip() for symbol in symbols if symbol})
    if not unique:
        return {}

    rows: list[dict[str, Any]] = []
    try:
        if len(unique) == 1:
            payload = _public_get(_TICKER_PATH, {"symbol": unique[0]})
            rows = [payload] if isinstance(payload, dict) else []
        else:
            payload = _public_get(
                _TICKER_PATH,
                {"symbols": json.dumps(unique, separators=(",", ":"))},
            )
            rows = payload if isinstance(payload, list) else []
    except Exception:
        rows = []

    result: dict[str, dict[str, float]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        try:
            last_price = float(row.get("lastPrice") or 0)
        except (TypeError, ValueError):
            continue
        if last_price <= 0:
            continue
        try:
            change_pct = float(row.get("priceChangePercent") or 0)
        except (TypeError, ValueError):
            change_pct = 0.0
        try:
            quote_volume = float(row.get("quoteVolume") or 0)
        except (TypeError, ValueError):
            quote_volume = 0.0
        result[symbol] = {
            "last_price": last_price,
            "price_change_pct": change_pct,
            "quote_volume": quote_volume,
        }

    missing = [symbol for symbol in unique if symbol not in result]
    if missing:
        try:
            if len(missing) == 1:
                price_payload = _public_get(_PRICE_PATH, {"symbol": missing[0]})
                price_rows = [price_payload] if isinstance(price_payload, dict) else []
            else:
                price_payload = _public_get(
                    _PRICE_PATH,
                    {"symbols": json.dumps(missing, separators=(",", ":"))},
                )
                price_rows = price_payload if isinstance(price_payload, list) else []
            for row in price_rows:
                symbol = str(row.get("symbol") or "").upper()
                try:
                    last_price = float(row.get("price") or 0)
                except (TypeError, ValueError):
                    continue
                if symbol and last_price > 0:
                    result[symbol] = {
                        "last_price": last_price,
                        "price_change_pct": 0.0,
                        "quote_volume": 0.0,
                    }
        except Exception:
            pass

    return result


def fetch_last_prices(symbols: list[str]) -> dict[str, float]:
    """Paritelerin anlık son fiyatlarını döndürür."""
    return {
        symbol: data["last_price"]
        for symbol, data in fetch_live_tickers(symbols).items()
        if data.get("last_price", 0) > 0
    }


def fetch_price_change_pct(symbol: str) -> Optional[float]:
    """
    Paritenin 24 saatlik fiyat değişim yüzdesini döndürür.

    Binance ``get_ticker`` → ``priceChangePercent`` alanı kullanılır.
    Hata durumunda ``None`` döner (strateji yine de çalışır).
    """
    try:
        ticker = _public_get(_TICKER_PATH, {"symbol": symbol.upper().strip()})
        if not ticker:
            _uyari(f"'{symbol}' için 24 saatlik ticker verisi boş döndü.")
            return None
        return float(ticker.get("priceChangePercent", 0))
    except Exception:
        return None
