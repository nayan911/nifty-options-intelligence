"""FYERS Instrument Master Provider and Symbol Resolution Engine.

Parses official NSE_FO.csv published by FYERS to resolve verified exchange contract symbols
without relying on synthetic string concatenation.
"""

from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
import pytz

from nifty_engine.core.config import settings
from nifty_engine.core.interfaces import IInstrumentMaster, OptionContract, OptionType

IST = pytz.timezone("Asia/Kolkata")

# Official 21-column schema of live https://public.fyers.in/sym_details/NSE_FO.csv
NSE_FO_COLUMNS = [
    "fytoken",
    "name",
    "instrument_type",
    "lot",
    "tick",
    "dummy_1",
    "trad_ses",
    "last_upd",
    "expiry_dt",
    "symbol",
    "exchange",
    "segment",
    "script_code",
    "short_sym",
    "underlying_id",
    "strike",
    "opt",
    "underlying_token",
    "reserved_1",
    "reserved_2",
    "reserved_3",
]


class FyersInstrumentMaster(IInstrumentMaster):
    """Downloads, caches, and indexes official FYERS instrument master files."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or settings.SYMBOL_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fo_cache_file = self.cache_dir / "NSE_FO.csv"

        self._df: Optional[pd.DataFrame] = None
        self._nifty_options_cache: Optional[pd.DataFrame] = None

    def load_master(self, fo_csv_path: Optional[str] = None) -> None:
        """Load instrument master into memory from a local CSV file."""
        path_to_load = Path(fo_csv_path) if fo_csv_path else self.fo_cache_file

        if not path_to_load.exists():
            raise FileNotFoundError(
                f"Symbol master not found at {path_to_load}. "
                "Download it using `download_master()` or provide a valid path."
            )

        # Load with pandas without headers using official 21 columns
        df = pd.read_csv(
            path_to_load,
            header=None,
            names=NSE_FO_COLUMNS,
            low_memory=False,
            dtype={
                "fytoken": str,
                "symbol": str,
                "short_sym": str,
                "strike": float,
                "opt": str,
                "lot": int,
                "tick": float,
                "expiry_dt": int,
            }
        )

        # Convert epoch expiry timestamp to local IST date
        df["expiry_date"] = pd.to_datetime(df["expiry_dt"], unit="s", utc=True).dt.tz_convert(IST).dt.date

        self._df = df
        # Pre-filter index options for fast queries
        self._nifty_options_cache = df[
            (df["short_sym"] == "NIFTY") & (df["opt"].isin(["CE", "PE"]))
        ].copy()

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Directly ingest a DataFrame for testing fixtures."""
        if "expiry_date" not in df.columns and "expiry_dt" in df.columns:
            df["expiry_date"] = pd.to_datetime(df["expiry_dt"], unit="s", utc=True).dt.tz_convert(IST).dt.date
        self._df = df
        self._nifty_options_cache = df[
            (df["short_sym"] == "NIFTY") & (df["opt"].isin(["CE", "PE"]))
        ].copy()

    def resolve_option(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        option_type: OptionType
    ) -> OptionContract:
        """Resolve the official FYERS symbol ticker for a specific contract."""
        if self._df is None:
            raise RuntimeError("Instrument master not loaded. Call load_master() first.")

        source_df = self._nifty_options_cache if underlying == "NIFTY" else self._df

        matches = source_df[
            (source_df["short_sym"] == underlying) &
            (source_df["expiry_date"] == expiry) &
            (source_df["strike"] == float(strike)) &
            (source_df["opt"] == option_type.value)
        ]

        if matches.empty:
            raise LookupError(
                f"No contract listed on exchange for {underlying} Expiry={expiry} Strike={strike} Type={option_type.value}"
            )

        row = matches.iloc[0]
        return OptionContract(
            symbol=row["symbol"],
            fytoken=str(row["fytoken"]),
            underlying=underlying,
            expiry=row["expiry_date"],
            strike=float(row["strike"]),
            option_type=OptionType(row["opt"]),
            lot_size=int(row["lot"]),
            tick_size=float(row["tick"]),
        )

    def resolve_future(
        self,
        underlying: str,
        expiry: Optional[date] = None
    ) -> str:
        """Resolve the official FYERS futures ticker for the underlying (e.g. NSE:NIFTY26NOVFUT)."""
        if self._df is None:
            raise RuntimeError("Instrument master not loaded. Call load_master() first.")

        today = date.today()
        futs = self._df[
            (self._df["short_sym"] == underlying) &
            (self._df["opt"] == "XX") &
            (self._df["expiry_date"] >= today)
        ].sort_values("expiry_date")

        if futs.empty:
            raise LookupError(f"No active futures contract found for {underlying}")

        if expiry is not None:
            matched = futs[futs["expiry_date"] == expiry]
            if matched.empty:
                raise LookupError(f"No futures contract found for {underlying} on {expiry}")
            return str(matched.iloc[0]["symbol"])

        # Return near-month future by default
        return str(futs.iloc[0]["symbol"])

    def get_atm_strike(self, underlying: str, spot_price: float) -> float:
        """Calculate the nearest At-The-Money (ATM) strike price based on index interval."""
        if underlying == "NIFTY":
            return round(spot_price / 50.0) * 50.0
        elif underlying == "BANKNIFTY":
            return round(spot_price / 100.0) * 100.0
        elif underlying == "SENSEX":
            return round(spot_price / 100.0) * 100.0
        return round(spot_price / 50.0) * 50.0

    def get_active_expiries(self, underlying: str) -> List[date]:
        """Return sorted list of future expiry dates currently listed for the underlying."""
        if self._df is None:
            raise RuntimeError("Instrument master not loaded. Call load_master() first.")

        source_df = self._nifty_options_cache if underlying == "NIFTY" else self._df
        today = date.today()

        future_expiries = source_df[
            (source_df["short_sym"] == underlying) &
            (source_df["expiry_date"] >= today)
        ]["expiry_date"].unique()

        return sorted(list(future_expiries))
