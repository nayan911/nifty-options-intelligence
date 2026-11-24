"""Stable decision codes for the live journal.

These strings are a public contract: journal rows, daily reports and research queries key on them.
Never rename or reuse a code; add new ones instead.
"""

from enum import Enum


class EntryDecision(str, Enum):
    ACCEPTED = "ACCEPTED"
    READINESS_NOT_READY = "READINESS_NOT_READY"
    MARKET_CLOSED = "MARKET_CLOSED"
    SPECIAL_SESSION_CUTOFF = "SPECIAL_SESSION_CUTOFF"
    BEFORE_MARKET_OPEN = "BEFORE_MARKET_OPEN"
    AFTER_ENTRY_CUTOFF = "AFTER_ENTRY_CUTOFF"
    NO_CONTRACT_SELECTED = "NO_CONTRACT_SELECTED"
    NET_RR_BELOW_MIN = "NET_RR_BELOW_MIN"
    STALE_OPTION_QUOTE = "STALE_OPTION_QUOTE"
    MAX_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"
    SAME_DIRECTION_OPEN = "SAME_DIRECTION_OPEN"
    SESSION_RISK_LIMIT = "SESSION_RISK_LIMIT"
    NOT_EVALUATED = "NOT_EVALUATED"   # candidate reached the journal without a paper-engine decision


class EventKind(str, Enum):
    SETUP = "SETUP"                # raw 5m/1m breakout-retest on NIFTY spot
    CANDIDATE = "CANDIDATE"        # multi-timeframe S000 candidate + its entry decision
    PAPER_FILL = "PAPER_FILL"
    PAPER_EXIT = "PAPER_EXIT"
