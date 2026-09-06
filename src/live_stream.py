"""
Canlı kline (mum) akış modülü.

Binance WebSocket üzerinden seçilen paritenin mum verisini dinler.
Her mum kapandığında geçmiş veriyi günceller ve strateji sinyalini üretir.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from binance import ThreadedWebsocketManager

from .data_engine import REQUEST_TIMEOUT, fetch_ohlcv, fetch_price_change_pct
from .strategy_engine import BaseStrategy, get_strategy, print_signal


@dataclass
class LiveStreamConfig:
    """Canlı akış yapılandırması."""

    symbol: str = "BTCUSDT"
    interval: str = "1h"
    history_limit: int = 500
    strategy_name: str = "futures_trend"


class LiveKlineStream:
    """
    Binance kline WebSocket dinleyicisi.

    Mum kapandığında (is_kline_closed == True) veri çeker ve sinyal üretir.
    """

    def __init__(
        self,
        config: LiveStreamConfig,
        strategy: Optional[BaseStrategy] = None,
    ) -> None:
        self.config = config
        self.strategy = strategy or get_strategy(config.strategy_name)

        self._twm: Optional[ThreadedWebsocketManager] = None
        self._conn_key: Optional[str] = None
        self._running = False
        self._lock = threading.Lock()
        # Aynı kapanan mum için tekrar işlem yapılmasını önler
        self._last_processed_open_time: Optional[int] = None

    def _on_kline_message(self, message: dict) -> None:
        """WebSocket'ten gelen kline mesajını işler."""
        # Bağlantı hatası veya beklenmeyen mesaj formatı
        if not isinstance(message, dict):
            return

        if message.get("e") != "kline":
            return

        kline = message.get("k", {})
        # Mum henüz kapanmadıysa bekle — yalnızca kapalı mumları işle
        if not kline.get("x"):
            return

        open_time = kline.get("t")
        if open_time is None:
            return

        with self._lock:
            if open_time == self._last_processed_open_time:
                return
            self._last_processed_open_time = open_time

        self._handle_closed_candle(kline)

    def _handle_closed_candle(self, kline: dict) -> None:
        """
        Kapanan mum sonrası geçmiş veriyi çeker ve strateji sinyalini üretir.

        Parameters
        ----------
        kline : dict
            WebSocket mesajındaki kline nesnesi.
        """
        symbol = self.config.symbol.upper()
        interval = self.config.interval

        print(
            f"\n[{self._timestamp()}] Mum kapandı — "
            f"{symbol} | {interval} | Kapanış: {kline.get('c', '—')}"
        )

        try:
            # Geçmiş OHLCV verisini REST API ile güncelle
            df = fetch_ohlcv(
                symbol=symbol,
                interval=interval,
                limit=self.config.history_limit,
            )
            if df is None or df.empty:
                print(
                    f"[{self._timestamp()}] Uyarı — {symbol} için mum verisi boş döndü, "
                    f"sinyal üretilmedi.",
                    file=sys.stderr,
                )
                return

            # Stratejiyi çalıştır ve sonucu terminale yazdır
            print_signal(
                df,
                symbol=symbol,
                interval=interval,
                strategy=self.strategy,
                price_change_pct_24h=fetch_price_change_pct(symbol),
            )
        except Exception as exc:
            print(
                f"[{self._timestamp()}] HATA — Sinyal üretilemedi: {exc}",
                file=sys.stderr,
            )

    @staticmethod
    def _timestamp() -> str:
        """Log satırları için okunabilir zaman damgası."""
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def start(self) -> None:
        """WebSocket bağlantısını başlatır ve ana döngüyü çalıştırır."""
        symbol = self.config.symbol.upper()
        interval = self.config.interval

        print("=" * 52)
        print("  Sinyalci — Canlı Kline Akışı")
        print("=" * 52)
        print(f"  Parite       : {symbol}")
        print(f"  Zaman dilimi : {interval}")
        print(f"  Strateji     : {self.config.strategy_name}")
        print(f"  Geçmiş mum   : {self.config.history_limit}")
        print("=" * 52)
        print("  Dinleniyor... (Durdurmak için Ctrl+C)")
        print()

        try:
            self._twm = ThreadedWebsocketManager(
                requests_params={"timeout": REQUEST_TIMEOUT},
            )
            self._twm.start()

            # Kline soketini başlat — her yeni mesaj _on_kline_message'a düşer
            self._conn_key = self._twm.start_kline_socket(
                callback=self._on_kline_message,
                symbol=symbol,
                interval=interval,
            )
        except Exception as exc:
            print(
                f"[{self._timestamp()}] Uyarı — Binance WebSocket başlatılamadı: {exc}",
                file=sys.stderr,
            )
            self.stop()
            return

        self._running = True
        self._register_shutdown_handlers()

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """WebSocket bağlantısını güvenli şekilde kapatır."""
        if not self._running and self._twm is None:
            return

        self._running = False

        if self._twm is not None:
            if self._conn_key is not None:
                self._twm.stop_socket(self._conn_key)
            self._twm.stop()
            self._twm = None

        print(f"\n[{self._timestamp()}] Canlı akış durduruldu.")

    def _register_shutdown_handlers(self) -> None:
        """Ctrl+C ve SIGTERM ile temiz kapanış sağlar."""

        def _shutdown_handler(signum, frame):
            del signum, frame
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown_handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _shutdown_handler)


def start_live_stream(
    config: LiveStreamConfig | None = None,
    strategy: Optional[BaseStrategy] = None,
) -> None:
    """
    Canlı kline akışını başlatır.

    Parameters
    ----------
    config : LiveStreamConfig, optional
        Parite, zaman dilimi ve strateji ayarları.
    strategy : BaseStrategy, optional
        Özel strateji örneği. Verilmezse config.strategy_name kullanılır.
    """
    stream = LiveKlineStream(config or LiveStreamConfig(), strategy=strategy)
    stream.start()
