"""DuckDB SQL Schemas for ticks, OHLCV candles, and option chain snapshots.

Adapted with attribution from marketcalls/ExpiryTrack database design patterns,
optimized for fast vectorized aggregations and time-bucket scans.
"""

CREATE_TABLES_SQL = """
-- High-frequency incoming ticks
CREATE TABLE IF NOT EXISTS ticks (
    symbol VARCHAR NOT NULL,
    exchange_timestamp TIMESTAMP WITH TIME ZONE,
    local_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    ltp DOUBLE NOT NULL,
    volume BIGINT NOT NULL,
    bid DOUBLE,
    ask DOUBLE,
    bid_qty BIGINT,
    ask_qty BIGINT,
    oi BIGINT,
    source VARCHAR DEFAULT 'FYERS',
    event_class VARCHAR DEFAULT 'LIVE'
);

-- Aggregated OHLCV Candlesticks (1m, 5m, 15m, 1d)
CREATE TABLE IF NOT EXISTS candles (
    symbol VARCHAR NOT NULL,
    timeframe VARCHAR NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume BIGINT NOT NULL,
    oi BIGINT,
    is_closed BOOLEAN DEFAULT FALSE,
    observed_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (symbol, timeframe, timestamp)
);

-- Option Chain Snapshots Header
CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    underlying VARCHAR NOT NULL,
    spot_price DOUBLE NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    expiry DATE NOT NULL,
    total_call_oi BIGINT,
    total_put_oi BIGINT,
    pcr DOUBLE
);

-- Option Chain Strike Details
CREATE TABLE IF NOT EXISTS option_chain_strikes (
    snapshot_id VARCHAR NOT NULL,
    strike DOUBLE NOT NULL,
    call_ltp DOUBLE,
    call_oi BIGINT,
    call_oi_change BIGINT,
    call_volume BIGINT,
    call_bid DOUBLE,
    call_ask DOUBLE,
    call_iv DOUBLE,
    call_delta DOUBLE,
    call_gamma DOUBLE,
    call_theta DOUBLE,
    call_vega DOUBLE,
    put_ltp DOUBLE,
    put_oi BIGINT,
    put_oi_change BIGINT,
    put_volume BIGINT,
    put_bid DOUBLE,
    put_ask DOUBLE,
    put_iv DOUBLE,
    put_delta DOUBLE,
    put_gamma DOUBLE,
    put_theta DOUBLE,
    put_vega DOUBLE,
    FOREIGN KEY (snapshot_id) REFERENCES option_chain_snapshots(snapshot_id)
);

-- Indexing for fast time-series queries
CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_ticks_lookup ON ticks(symbol, local_timestamp);
"""
