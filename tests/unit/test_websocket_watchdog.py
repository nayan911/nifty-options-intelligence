"""Unit tests for the FYERS WebSocket liveness watchdog (silent-stream reconnect)."""

from unittest.mock import MagicMock, patch

import pytest

from nifty_engine.data.fyers import websocket_client as wc
from nifty_engine.data.fyers.websocket_client import FyersWebSocketClient


class FakeSocket:
    """Stand-in for FyersDataSocket capturing callbacks and subscribe calls."""

    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.restart_flag = True
        self.subscribed = []
        self._FyersDataSocket__ws_object = MagicMock()
        self.closed = False
        FakeSocket.instances.append(self)

    def connect(self):
        pass

    def subscribe(self, symbols, data_type="SymbolUpdate"):
        self.subscribed.append(list(symbols))

    def close_connection(self):
        self.closed = True

    # helpers to simulate SDK events
    def fire_open(self):
        self.kwargs["on_connect"]()

    def fire_message(self, msg):
        self.kwargs["on_message"](msg)

    def fire_close(self):
        self.kwargs["on_close"]("closed")


@pytest.fixture
def client():
    FakeSocket.instances = []
    auth = MagicMock()
    auth.load_cached_token.return_value = "tok"
    ticks = []
    c = FyersWebSocketClient(auth_provider=auth, on_tick=ticks.append)
    c._ticks = ticks
    with patch.object(wc, "FyersDataSocket", FakeSocket):
        yield c


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _quote(sym="NSE:NIFTY50-INDEX", ltp=25000.0):
    return {"symbol": sym, "ltp": ltp, "exch_feed_time": 1795687200}


def test_healthy_stream_takes_no_action(client):
    clock = Clock()
    with patch.object(wc.time, "monotonic", clock):
        client.connect()
        FakeSocket.instances[0].fire_open()
        clock.t += 10
        FakeSocket.instances[0].fire_message(_quote())
        clock.t += 10
        assert client.ensure_alive(stale_after=15.0) is None
    assert len(FakeSocket.instances) == 1


def test_silent_but_connected_stream_forces_reconnect_and_resubscribe(client):
    """Wi-Fi drop: SDK still thinks it's connected, but no data arrives."""
    clock = Clock()
    with patch.object(wc.time, "monotonic", clock):
        client.subscribe(["NSE:NIFTY50-INDEX", "NSE:NIFTY26NOVFUT"])
        client.connect()
        old = FakeSocket.instances[0]
        old.fire_open()
        assert client.is_connected

        clock.t += 20  # silence beyond threshold
        action = client.ensure_alive(stale_after=15.0)
        assert action and "silent" in action

        assert len(FakeSocket.instances) == 2
        assert old.restart_flag is False  # SDK's own reconnect loop stopped

        new = FakeSocket.instances[1]
        new.fire_open()
        assert client.is_connected
        assert sorted(new.subscribed[-1]) == ["NSE:NIFTY26NOVFUT", "NSE:NIFTY50-INDEX"]
        assert client.reconnect_count == 1


def test_stale_callbacks_from_abandoned_socket_are_ignored(client):
    clock = Clock()
    with patch.object(wc.time, "monotonic", clock):
        client.connect()
        old = FakeSocket.instances[0]
        old.fire_open()
        clock.t += 20
        client.ensure_alive(stale_after=15.0)
        new = FakeSocket.instances[1]
        new.fire_open()

        # Late close/tick from the dead socket must not flip state or inject data
        old.fire_close()
        assert client.is_connected
        old.fire_message(_quote(ltp=1.0))
        assert client._ticks == []

        new.fire_message(_quote(ltp=25010.0))
        assert [t.ltp for t in client._ticks] == [25010.0]


def test_reconnect_backoff_when_network_still_down(client):
    clock = Clock()
    with patch.object(wc.time, "monotonic", clock):
        client.connect()
        FakeSocket.instances[0].fire_open()
        clock.t += 20

        # Network is down: constructing/connecting the new socket raises
        with patch.object(FakeSocket, "connect", side_effect=OSError("no route")):
            first = client.ensure_alive(stale_after=15.0)
            assert "failed" in first
            # Immediately after, watchdog waits for the backoff window
            clock.t += 1
            assert client.ensure_alive(stale_after=15.0) is None
            clock.t += 10
            second = client.ensure_alive(stale_after=15.0)
            assert "failed" in second

        # Network back: next attempt after backoff succeeds and data resets failures
        clock.t += 60
        ok = client.ensure_alive(stale_after=15.0)
        assert ok and "reconnecting" in ok
        FakeSocket.instances[-1].fire_open()
        FakeSocket.instances[-1].fire_message(_quote())
        assert client._reconnect_failures == 0


def test_close_does_not_hang_on_stuck_sdk(client):
    client.connect()
    sock = FakeSocket.instances[0]

    import threading

    block = threading.Event()
    sock.close_connection = lambda: block.wait(30)  # simulate SDK join() hang
    client.close(timeout=0.2)
    assert not client.is_connected
    block.set()
