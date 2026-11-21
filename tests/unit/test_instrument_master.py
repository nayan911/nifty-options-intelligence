"""Unit tests for FyersInstrumentMaster and official symbol resolution."""

from datetime import date

import pandas as pd
import pytest

from nifty_engine.core.interfaces import OptionType
from nifty_engine.data.fyers.instrument_master import NSE_FO_COLUMNS, FyersInstrumentMaster


@pytest.fixture
def mock_instrument_master():
    master = FyersInstrumentMaster()

    # Exact epoch for 2026-11-26 15:30:00 IST is 1795687200
    expiry_epoch = 1795687200

    # 21 columns matching official live NSE_FO.csv layout
    rows = [
        # NIFTY 25000 CE (expiry 2026-11-26)
        [
            "101126112625000CE", "NIFTY 26 Nov 25000 CE", 14, 65, 0.05, "",
            "0915-1540|1815-1915:", "2026-11-20", expiry_epoch, "NSE:NIFTY26NOV25000CE",
            10, 11, 15001, "NIFTY", 26000, 25000.0, "CE", "101000000026000", "None", 0, 0.0
        ],
        # NIFTY 25000 PE
        [
            "101126112625000PE", "NIFTY 26 Nov 25000 PE", 14, 65, 0.05, "",
            "0915-1540|1815-1915:", "2026-11-20", expiry_epoch, "NSE:NIFTY26NOV25000PE",
            10, 11, 15002, "NIFTY", 26000, 25000.0, "PE", "101000000026000", "None", 0, 0.0
        ],
        # NIFTY 25050 CE
        [
            "101126112625050CE", "NIFTY 26 Nov 25050 CE", 14, 65, 0.05, "",
            "0915-1540|1815-1915:", "2026-11-20", expiry_epoch, "NSE:NIFTY26NOV25050CE",
            10, 11, 15003, "NIFTY", 26000, 25050.0, "CE", "101000000026000", "None", 0, 0.0
        ],
        # BANKNIFTY 52000 CE
        [
            "101126112652000CE", "BANKNIFTY 26 Nov 52000 CE", 14, 30, 0.05, "",
            "0915-1540|1815-1915:", "2026-11-20", expiry_epoch, "NSE:BANKNIFTY26NOV52000CE",
            10, 11, 15004, "BANKNIFTY", 26009, 52000.0, "CE", "101000000026009", "None", 0, 0.0
        ],
        # NIFTY near-month FUT
        [
            "1011261126FUT", "NIFTY 26 Nov FUT", 14, 65, 0.1, "",
            "0915-1540|1815-1915:", "2026-11-20", expiry_epoch, "NSE:NIFTY26NOVFUT",
            10, 11, 15005, "NIFTY", 26000, 0.0, "XX", "101000000026000", "None", 0, 0.0
        ],
    ]

    df = pd.DataFrame(rows, columns=NSE_FO_COLUMNS)
    master.load_from_dataframe(df)
    return master


def test_resolve_official_symbol(mock_instrument_master):
    expiry_date = date(2026, 11, 26)

    # 1. Resolve 25000 CE
    contract_ce = mock_instrument_master.resolve_option(
        underlying="NIFTY",
        expiry=expiry_date,
        strike=25000.0,
        option_type=OptionType.CE
    )
    assert contract_ce.symbol == "NSE:NIFTY26NOV25000CE"
    assert contract_ce.lot_size == 65
    assert contract_ce.tick_size == 0.05

    # 2. Resolve 25000 PE
    contract_pe = mock_instrument_master.resolve_option(
        underlying="NIFTY",
        expiry=expiry_date,
        strike=25000.0,
        option_type=OptionType.PE
    )
    assert contract_pe.symbol == "NSE:NIFTY26NOV25000PE"
    assert contract_pe.lot_size == 65


def test_atm_strike_calculation(mock_instrument_master):
    # NIFTY interval = 50.0
    assert mock_instrument_master.get_atm_strike("NIFTY", 25023.40) == 25000.0
    assert mock_instrument_master.get_atm_strike("NIFTY", 25026.00) == 25050.0
    assert mock_instrument_master.get_atm_strike("NIFTY", 25075.00) == 25100.0

    # BANKNIFTY interval = 100.0
    assert mock_instrument_master.get_atm_strike("BANKNIFTY", 52040.0) == 52000.0
    assert mock_instrument_master.get_atm_strike("BANKNIFTY", 52060.0) == 52100.0


def test_unlisted_contract_raises_lookup_error(mock_instrument_master):
    with pytest.raises(LookupError) as exc_info:
        mock_instrument_master.resolve_option(
            underlying="NIFTY",
            expiry=date(2026, 11, 26),
            strike=29999.0,  # Unlisted strike
            option_type=OptionType.CE
        )
    assert "No contract listed" in str(exc_info.value)


def test_resolve_future(mock_instrument_master):
    fut_symbol = mock_instrument_master.resolve_future("NIFTY")
    assert fut_symbol == "NSE:NIFTY26NOVFUT"
