"""Live journal recorder: called from the market-data runner's callbacks. OBSERVE-ONLY.

Every method swallows and logs its own errors: the journal must never interrupt or alter trading.
Indicators are not computed here: they are computed at end of day from the stored bars with the same
causal library (values at minute m use only bars closed by m), so they equal what was knowable live.
What is recorded here is what only exists live: the decision, readiness, feed freshness and the
option quote at the signal.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional

from nifty_engine.journal.codes import EntryDecision, EventKind
from nifty_engine.journal.store import JournalStore
from nifty_engine.utils.timezone import now_ist, to_ist

logger = logging.getLogger(__name__)

TRACK_MINUTES = 60


class JournalRecorder:
    def __init__(self, journal: JournalStore, strategy: Any, clock: Callable[[], datetime] = now_ist) -> None:
        self.journal = journal
        self.strategy = strategy
        self.clock = clock

    def _base(self, kind: EventKind, event_at: datetime, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        at = to_ist(event_at)
        ctx = context or {}
        return {
            "kind": kind.value, "event_at": at, "day": at.date(), "recorded_at": self.clock(),
            "strategy_id": self.strategy.id, "strategy_version": self.strategy.version,
            "strategy_fingerprint": self.strategy.fingerprint(),
            "readiness": ctx.get("readiness"), "worst_feed_symbol": ctx.get("worst_feed_symbol"),
            "worst_feed_age_s": ctx.get("worst_feed_age_s"),
        }

    def on_setup(self, setup: Any, snap: Any, context: Optional[Dict[str, Any]] = None) -> None:
        try:
            at = getattr(snap, "effective_available_at", None) or snap.timestamp
            row = self._base(EventKind.SETUP, at, context)
            row.update({
                "event_id": f"SETUP|{snap.symbol}|{snap.timeframe}|{snap.timestamp.isoformat()}|"
                            f"{setup.direction.value}|{setup.level_id}",
                "symbol": snap.symbol, "timeframe": snap.timeframe, "direction": setup.direction.value,
                "level_id": setup.level_id,
                "payload": {"bar_at": snap.timestamp, "level": setup.nominal_price, "stop": setup.stop_loss,
                            "target": setup.target_price, "close": snap.close, "trend": snap.trend_bias},
            })
            self.journal.add_event(row)
        except Exception as e:  # never interrupt trading
            logger.warning("journal: failed to record setup: %s", e)

    def on_candidate(self, candidate: Any, decision: Optional[Dict[str, Any]],
                     context: Optional[Dict[str, Any]] = None) -> None:
        try:
            dec = decision if decision and decision.get("candidate_id") == candidate.candidate_id else None
            at = candidate.decision_at or candidate.timestamp
            row = self._base(EventKind.CANDIDATE, at, context)
            da = getattr(candidate, "derivatives_assessment", None)
            sc = getattr(da, "selected_contract", None) if da is not None else None
            row.update({
                "event_id": f"CANDIDATE|{candidate.candidate_id}",
                "symbol": candidate.symbol, "timeframe": "1m", "direction": candidate.direction.value,
                "level_id": candidate.setup_level, "candidate_id": candidate.candidate_id,
                "decision_code": (dec or {}).get("code", EntryDecision.NOT_EVALUATED.value),
                "decision_detail": (dec or {}).get("detail", ""),
                "payload": {
                    "bar_at": candidate.timestamp, "decision_at": candidate.decision_at,
                    "entry": candidate.entry_price, "stop": candidate.stop_loss, "target_1": candidate.target_1,
                    "target_2": candidate.target_2, "risk_points": candidate.risk_points,
                    "confluence": candidate.confluence_score, "headroom": candidate.headroom_points,
                    "confirmation": candidate.confirmation_status, "breakdown": candidate.timeframe_breakdown,
                    "derivatives": (None if da is None else {
                        "assessment": getattr(getattr(da, "assessment_type", None), "value", None),
                        "alignment": getattr(da, "alignment_score", None),
                        "contract": getattr(sc, "symbol", None), "ltp": getattr(sc, "ltp", None),
                        "bid": getattr(sc, "bid", None), "ask": getattr(sc, "ask", None),
                        "spread_pct": getattr(sc, "spread_pct", None),
                    }),
                    "decision": dec,
                },
            })
            self.journal.add_event(row)
        except Exception as e:
            logger.warning("journal: failed to record candidate: %s", e)

    def on_trade_event(self, ev: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> None:
        try:
            kind = {"ORDER_FILLED": EventKind.PAPER_FILL, "POSITION_CLOSED": EventKind.PAPER_EXIT}.get(ev.get("event"))
            if kind is None:
                return
            at = self.clock()
            row = self._base(kind, at, context)
            row.update({
                "event_id": f"{kind.value}|{ev.get('position_id')}|{ev.get('reason') or ''}|{at.isoformat()}",
                "symbol": ev.get("symbol"), "decision_detail": ev.get("reason"), "payload": ev,
            })
            self.journal.add_event(row)
        except Exception as e:
            logger.warning("journal: failed to record trade event: %s", e)

    def track_option(self, candidate: Any, contract: Optional[str], quote: Optional[Dict[str, Any]],
                     source: str) -> Optional[str]:
        """Pin `contract` for premium tracking for TRACK_MINUTES after the decision. Returns the contract."""
        try:
            if not contract:
                return None
            at = to_ist(candidate.decision_at or candidate.timestamp)
            self.journal.add_tracking(candidate.candidate_id, at.date(), contract, candidate.direction.value, at,
                                      at + timedelta(minutes=TRACK_MINUTES), source, quote)
            return contract
        except Exception as e:
            logger.warning("journal: failed to start option tracking: %s", e)
            return None
