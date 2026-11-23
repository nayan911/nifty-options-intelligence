"""Strategy registry loader (strategies/registry.yaml), validated strictly.

Any problem (bad YAML, unknown or missing keys, wrong types, out-of-range or contradictory values,
FIXED definitions that differ from the engine) raises RegistryError, so neither the live pipeline
nor a backtest can silently run with parameters other than the ones written down.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nifty_engine.strategy.params import S000_FIXED, S000Params

DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "strategies" / "registry.yaml"
# Version locks: {strategy_id: {version: fingerprint}}. The fingerprint covers everything that changes
# behaviour or which data a result may come from (instruments, data requirements incl. contract
# confidence policy and warmup, params, fixed definitions) - not name/description/evidence/changelog.
# Once a version is established it is frozen; any such change needs a new version or strategy id.
DEFAULT_LOCK = DEFAULT_REGISTRY.with_name("registry.lock.json")
LOCK_REQUIRED_STATUSES = ("baseline", "observe", "live")
SCHEMA_VERSION = 1
CONFIDENCE_TIERS = ("VERIFIED", "PARTIALLY_VERIFIED", "INFERRED")

# Strategy id -> (params model, fixed engine definitions). New strategies register here.
PARAM_MODELS = {"S000": (S000Params, S000_FIXED)}


class RegistryError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SeriesRequirement(_Strict):
    instrument: str
    kind: Literal["SPOT", "CONT", "CONT_NEXT", "OWN"]
    resolutions: List[str] = Field(min_length=1)


class DataRequirements(_Strict):
    series: List[SeriesRequirement] = Field(min_length=1)
    contract_confidence: List[str] = Field(min_length=1)
    warmup_trading_days: int = Field(ge=1, le=60)
    # signal_and_warmup: the signal day AND the warmup days before it must all be accepted tiers
    # (an input from a rejected tier can reach the signal through warmup indicators/levels).
    # signal_day: only the signal day is checked.
    contract_confidence_scope: Literal["signal_and_warmup", "signal_day"]

    @field_validator("contract_confidence")
    @classmethod
    def _tiers(cls, v):
        bad = [t for t in v if t not in CONFIDENCE_TIERS]
        if bad:
            raise ValueError(f"unknown contract confidence tier(s) {bad}; allowed {list(CONFIDENCE_TIERS)}"
                             " (CONFLICT can never be accepted)")
        return v


class Instruments(_Strict):
    underlying: str
    signal: str
    confirmation: str


class ChangelogEntry(_Strict):
    version: int = Field(ge=1)
    date: date
    change: str = Field(min_length=1)


class StrategyEntry(_Strict):
    id: str
    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    status: Literal["baseline", "research", "observe", "live", "retired"]
    description: str = Field(min_length=1)
    instruments: Instruments
    data_requirements: DataRequirements
    params: Any
    fixed: Dict[str, Any]
    evidence: Optional[Dict[str, Any]] = None
    changelog: List[ChangelogEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self):
        if self.changelog[-1].version != self.version:
            raise ValueError(f"changelog must end with an entry for version {self.version}")
        return self

    def params_hash(self) -> str:
        """Hash of the engine parameters + fixed definitions only (what the technical engine runs)."""
        payload = {"params": self.params.canonical(), "fixed": self.fixed}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def fingerprint(self) -> str:
        """Immutable identity of this strategy version (what registry.lock.json locks): instruments,
        data requirements (series, contract-confidence policy and scope, warmup), params, fixed."""
        payload = {
            "instruments": self.instruments.model_dump(mode="json"),
            "data_requirements": self.data_requirements.model_dump(mode="json"),
            "params": self.params.canonical(),
            "fixed": self.fixed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def provenance(self) -> Dict[str, Any]:
        return {"id": self.id, "version": self.version, "status": self.status,
                "fingerprint": self.fingerprint(), "params_hash": self.params_hash()}


class Registry(_Strict):
    schema_version: int
    strategies: Dict[str, StrategyEntry]
    path: Optional[str] = None

    def get(self, strategy_id: str) -> StrategyEntry:
        if strategy_id not in self.strategies:
            raise RegistryError(f"strategy {strategy_id!r} is not in the registry ({sorted(self.strategies)})")
        return self.strategies[strategy_id]


def _parse_strategy(sid: str, raw: Any) -> StrategyEntry:
    if sid not in PARAM_MODELS:
        raise RegistryError(f"{sid}: no parameter model is registered in code for this strategy id")
    if not isinstance(raw, dict):
        raise RegistryError(f"{sid}: entry must be a mapping")
    model, fixed_expected = PARAM_MODELS[sid]
    if "params" not in raw:
        raise RegistryError(f"{sid}: missing 'params'")
    # Every parameter must be written explicitly in the file (no silent defaults).
    missing = _missing_keys(model, raw["params"])
    if missing:
        raise RegistryError(f"{sid}: params missing explicit value(s): {missing}")
    try:
        # Strict JSON-mode validation: no coercion of strings ("1.3"), booleans or floats-as-ints.
        # (JSON mode still accepts YAML lists for tuples and ints for float fields.)
        params = model.model_validate_json(_to_json(raw["params"]), strict=True)
        meta = {k: v for k, v in raw.items() if k != "params"}
        entry = StrategyEntry.model_validate_json(_to_json({**meta, "id": sid, "params": None}), strict=True)
        entry = entry.model_copy(update={"params": params})
    except ValidationError as e:
        raise RegistryError(f"{sid}: invalid registry entry:\n{e}") from None
    if _normalize(entry.fixed) != _normalize(fixed_expected):
        raise RegistryError(
            f"{sid}: 'fixed' does not match the engine's definitions.\n  registry: {entry.fixed}\n"
            f"  engine:   {fixed_expected}\nThese are computed in code; change them only with a code change "
            "and a new strategy."
        )
    return entry


def _missing_keys(model, raw: Any, prefix: str = "") -> List[str]:
    out: List[str] = []
    raw = raw if isinstance(raw, dict) else {}
    for name, field in model.model_fields.items():
        key = field.alias or name
        path = f"{prefix}{key}"
        if key not in raw:
            out.append(path)
            continue
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out.extend(_missing_keys(ann, raw[key], path + "."))
    return out


def _to_json(obj: Any) -> str:
    def default(o):
        if isinstance(o, date):
            return o.isoformat()
        raise TypeError(f"unsupported value {o!r} ({type(o).__name__})")
    return json.dumps(obj, default=default)


def _check_locks(entries: Dict[str, StrategyEntry], lock_path: Path) -> None:
    try:
        locks = json.loads(lock_path.read_text())
    except FileNotFoundError:
        raise RegistryError(f"strategy version lock file not found: {lock_path}") from None
    except json.JSONDecodeError as e:
        raise RegistryError(f"strategy version lock file {lock_path} is not valid JSON: {e}") from None
    for sid, e in entries.items():
        locked = locks.get(sid, {}).get(str(e.version))
        h = e.fingerprint()
        if locked is not None and locked != h:
            raise RegistryError(
                f"{sid} v{e.version} is locked to fingerprint {locked}, but the registry now fingerprints to {h}. "
                f"Behaviour or data policy changed: give it a new version (or a new strategy id) with a "
                f"changelog entry, and lock that version instead of editing v{e.version}."
            )
        if locked is None and e.status in LOCK_REQUIRED_STATUSES:
            raise RegistryError(
                f"{sid} v{e.version} has status {e.status!r} but no version lock; add "
                f'"{sid}": {{"{e.version}": "{h}"}} to {lock_path.name} once this version is established.'
            )


def _normalize(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def load_registry(path: Union[str, Path, None] = None, lock_path: Union[str, Path, None] = None) -> Registry:
    """lock_path defaults to registry.lock.json next to the registry file."""
    p = Path(path) if path is not None else DEFAULT_REGISTRY
    lp = Path(lock_path) if lock_path is not None else p.with_name(DEFAULT_LOCK.name)
    try:
        raw = yaml.safe_load(p.read_text())
    except FileNotFoundError:
        raise RegistryError(f"strategy registry not found: {p}") from None
    except yaml.YAMLError as e:
        raise RegistryError(f"strategy registry {p} is not valid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise RegistryError(f"strategy registry {p} must be a mapping")
    extra = set(raw) - {"schema_version", "strategies"}
    if extra:
        raise RegistryError(f"unknown top-level key(s) {sorted(extra)}")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError(f"schema_version must be {SCHEMA_VERSION}, got {raw.get('schema_version')!r}")
    strategies = raw.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        raise RegistryError("'strategies' must be a non-empty mapping")
    entries = {str(sid): _parse_strategy(str(sid), body) for sid, body in strategies.items()}
    _check_locks(entries, lp)
    return Registry(schema_version=SCHEMA_VERSION, strategies=entries, path=str(p))
