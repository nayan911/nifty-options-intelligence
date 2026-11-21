"""Timezone utilities for Indian Standard Time (IST, UTC+5:30)."""

from datetime import datetime, timezone
import pytz

IST = pytz.timezone("Asia/Kolkata")


def now_ist() -> datetime:
    """Return the current time in IST timezone."""
    return datetime.now(IST)


def to_ist(dt: datetime) -> datetime:
    """Convert any naive or aware datetime to IST."""
    if dt.tzinfo is None:
        return IST.localize(dt)
    return dt.astimezone(IST)


def epoch_to_ist(epoch_seconds: float) -> datetime:
    """Convert a Unix epoch timestamp to an aware IST datetime."""
    return datetime.fromtimestamp(epoch_seconds, tz=IST)
