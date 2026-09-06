"""
Veri motoru modülü.

Binance borsasından geçmiş OHLCV (Open, High, Low, Close, Volume) mum
verilerini çeker, temizler ve analiz için hazır bir Pandas DataFrame döndürür.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from binance.client import Client


# Binance API istemcisi — geçmiş mum verisi public endpoint olduğu için
# API anahtarı olmadan da kullanılabilir.
_client: Optional[Client] = None


def _get_client() -> Client:
    """Tekil (singleton) Binance istemcisini döndürür."""
    global _client
    if _client is None:
        _client = Client()
    return _client


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

    Raises
    ------
    ValueError
        Geçersiz parametre veya boş veri döndüğünde.
    """
    # Parite sembolünü Binance formatına uygun hale getir (büyük harf)
    symbol = symbol.upper().strip()
    interval = interval.strip()

    if not symbol:
        raise ValueError("Parite (symbol) boş olamaz.")
    if not interval:
        raise ValueError("Zaman dilimi (interval) boş olamaz.")

    client = _get_client()

    # Başlangıç zamanı verilmişse tarih aralığına göre, aksi halde son N muma göre çek
    if start_time:
        raw_klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time,
            end_str=end_time,
        )
    else:
        # Limit değerini Binance'in izin verdiği aralıkta tut
        safe_limit = max(1, min(limit, 1000))
        raw_klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=safe_limit,
        )

    if not raw_klines:
        raise ValueError(
            f"'{symbol}' paritesi için '{interval}' diliminde veri bulunamadı."
        )

    return _clean_klines(raw_klines)


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


def fetch_price_change_pct(symbol: str) -> Optional[float]:
    """
    Paritenin 24 saatlik fiyat değişim yüzdesini döndürür.

    Binance ``get_ticker`` → ``priceChangePercent`` alanı kullanılır.
    Hata durumunda ``None`` döner (strateji yine de çalışır).
    """
    try:
        client = _get_client()
        ticker = client.get_ticker(symbol=symbol.upper())
        return float(ticker.get("priceChangePercent", 0))
    except Exception:
        return None
