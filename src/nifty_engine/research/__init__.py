"""Research layer: historical datasets, indicator features, backtests and the strategy registry.

Kept separate from the live pipeline (own DuckDB file) so research jobs never contend for the
live database lock or pollute live warmup history.
"""
