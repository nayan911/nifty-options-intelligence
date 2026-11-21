"""Unit tests for FyersWebSocketClient quote and depth parsing."""

from unittest.mock import MagicMock

from nifty_engine.data.fyers.websocket_client import FyersWebSocketClient


def test_websocket_quote_parsing_with_bid_ask_depth():
    """Verify that FYERS WebSocket message parsing extracts bid/ask prices and quantities."""
    auth_mock = MagicMock()
    received_ticks = []

    client = FyersWebSocketClient(
        auth_provider=auth_mock,
        on_tick=lambda t: received_ticks.append(t),
    )

    # Message with full depth and FYERS fields
    msg = {
        "symbol": "NSE:NIFTY26NOVFUT",
        "ltp": 25050.25,
        "vol_traded_today": 125000,
        "exch_feed_time": 1795687200,
        "bid": 25050.0,
        "ask": 25050.5,
        "bid_qty": 450,
        "ask_qty": 600,
        "oi": 850000,
    }

    client._on_message(msg)

    assert len(received_ticks) == 1
    t = received_ticks[0]
    assert t.symbol == "NSE:NIFTY26NOVFUT"
    assert t.ltp == 25050.25
    assert t.cumulative_volume == 125000
    assert t.bid == 25050.0
    assert t.ask == 25050.5
    assert t.bid_qty == 450
    assert t.ask_qty == 600
    assert t.oi == 850000


def test_websocket_quote_parsing_field_variations():
    """Verify that alternate FYERS field names (bid_price, ask_price, b_qty, a_qty) are parsed."""
    auth_mock = MagicMock()
    received_ticks = []

    client = FyersWebSocketClient(
        auth_provider=auth_mock,
        on_tick=lambda t: received_ticks.append(t),
    )

    msg = {
        "symbol": "NSE:NIFTY26NOV25000CE",
        "ltp": 120.5,
        "vol_traded_today": 50000,
        "exch_feed_time": 1795687200,
        "bid_price": 120.0,
        "ask_price": 120.5,
        "b_qty": 325,
        "a_qty": 195,
        "oi": 1200000,
    }

    client._on_message(msg)

    assert len(received_ticks) == 1
    t = received_ticks[0]
    assert t.symbol == "NSE:NIFTY26NOV25000CE"
    assert t.bid == 120.0
    assert t.ask == 120.5
    assert t.bid_qty == 325
    assert t.ask_qty == 195


def test_websocket_ack_messages_handled():
    """Verify that sub/auth system messages increment ack counters and invoke on_ack."""
    auth_mock = MagicMock()
    received_acks = []

    client = FyersWebSocketClient(
        auth_provider=auth_mock,
        on_ack=lambda a: received_acks.append(a),
    )

    client._on_message({"type": "sub", "message": "Subscribed successfully"})
    assert client.subscription_acks == 1
    assert len(received_acks) == 1
