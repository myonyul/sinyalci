"""
Sinyalci — Ana giriş noktası.

Canlı Binance kline akışını başlatır; her mum kapandığında
strateji sinyalini terminale yazdırır.

Kullanım:
    python main.py
    python main.py --symbol ETHUSDT --interval 15m
    python main.py --symbol BTCUSDT --interval 1h --limit 300
"""

from __future__ import annotations

import argparse
import sys

from src.live_stream import LiveStreamConfig, start_live_stream


def _build_parser() -> argparse.ArgumentParser:
    """Komut satırı argümanlarını tanımlar."""
    parser = argparse.ArgumentParser(
        description="Sinyalci — Canlı kripto al-sat sinyal botu",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="İşlem çifti (varsayılan: BTCUSDT)",
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Mum zaman dilimi: 1m, 15m, 1h, 4h, 1d vb. (varsayılan: 1h)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Strateji için çekilecek geçmiş mum sayısı (varsayılan: 500)",
    )
    parser.add_argument(
        "--strategy",
        default="futures_trend",
        help="Kullanılacak strateji adı (varsayılan: futures_trend)",
    )
    return parser


def main() -> None:
    """Uygulamayı başlatır."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.limit < 1 or args.limit > 1000:
        print("HATA: --limit değeri 1 ile 1000 arasında olmalıdır.", file=sys.stderr)
        sys.exit(1)

    config = LiveStreamConfig(
        symbol=args.symbol,
        interval=args.interval,
        history_limit=args.limit,
        strategy_name=args.strategy,
    )

    start_live_stream(config)


if __name__ == "__main__":
    main()
