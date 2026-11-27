"""journal_day --refresh window: must cover every 1m bar already in the accumulated daily history DB."""

from datetime import date

import duckdb
from journal_day import WINDOW_DAYS, refresh_start


def _make_db(path, days):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE hist_bars (ts TIMESTAMP, resolution VARCHAR)")
    for d in days:
        con.execute("INSERT INTO hist_bars VALUES (?, '1')", [f"{d} 09:15:00"])
        con.execute("INSERT INTO hist_bars VALUES (?, 'D')", ["2020-01-01 00:00:00"])  # daily bars do not count
    con.close()


def test_no_db_uses_default_window(tmp_path):
    assert refresh_start(date(2026, 11, 26), tmp_path / "missing.duckdb") == date(2026, 10, 27)


def test_window_extends_back_to_earliest_existing_bar(tmp_path):
    # An earlier refresh for 25-Nov downloaded from 26-Oct; refreshing 26-Nov must still cover 26-Oct,
    # otherwise the replay rejects the manifest ("Manifest covers 2026-10-27.., backtest needs 2026-10-26..").
    db = tmp_path / "history.duckdb"
    _make_db(db, ["2026-10-26", "2026-11-25"])
    assert refresh_start(date(2026, 11, 26), db) == date(2026, 10, 26)


def test_recent_db_keeps_default_window(tmp_path):
    db = tmp_path / "history.duckdb"
    _make_db(db, ["2026-11-20"])
    day = date(2026, 11, 26)
    assert (day - refresh_start(day, db)).days == WINDOW_DAYS
