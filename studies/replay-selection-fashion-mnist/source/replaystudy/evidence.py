"""Append-first experiment retention and independent, read-only verification.

This module does not import the learner, runner, data loader, or scorer. A COMPLETE
marker is not proof: ``verify_bundle`` rechecks identities, schemas, provenance,
checkpoint bytes, and a full artifact inventory. Failed records are retained.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile
from typing import Any, Mapping, Sequence

import numpy as np

ARMS = ("mir", "highloss", "uniform", "shuffledscore", "uniform-extra-replay")
ETA_GRID = (0.03, 0.1, 0.3)
PROVENANCE_REQUIRED = {"source", "tests", "protocol", "data", "indices"}
MARKERS = {"RUNNING", "FAILED", "INTERRUPTED", "COMPLETE"}
_RESERVED = MARKERS | {"artifacts.json"}
_SCHEMA = {
    "software": {"metrics.validation_recall": [10]},
    "development": {"metrics.validation_matrix": [5, 5],
                    "metrics.validation_recall": [10]},
    "evaluation": {"metrics.test_matrix": [5, 5], "metrics.full_test_matrix": [5, 5],
                   "metrics.test_recall": [10], "metrics.full_test_recall": [10]},
}


class EvidenceError(ValueError):
    """Evidence was retained but cannot form a valid completed experiment."""


class VerificationError(EvidenceError):
    """Independent inspection found an invalid or incomplete evidence bundle."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_case_id(seed: int, arm: str, eta: float) -> str:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}")
    if isinstance(eta, bool) or not isinstance(eta, (int, float, np.number)):
        raise ValueError("eta must be a finite positive number")
    if not math.isfinite(float(eta)) or float(eta) <= 0:
        raise ValueError("eta must be a finite positive number")
    return f"seed-{int(seed)}__{arm}__eta-{float(eta):.12g}"


def expected_cases(stage: str, *, diagnostic_eta: float | None = None,
                   extra_eta: float | None = None) -> list[dict[str, Any]]:
    """Return the prospective exact research case grid, in deterministic order."""
    if stage == "development":
        triples = ((seed, arm, eta) for seed in (1001, 1002)
                   for eta in ETA_GRID for arm in ARMS)
    elif stage == "evaluation":
        if diagnostic_eta not in ETA_GRID or extra_eta not in ETA_GRID:
            raise ValueError("evaluation needs both selected rates from the declared grid")
        triples = ((seed, arm, extra_eta if arm == "uniform-extra-replay" else diagnostic_eta)
                   for seed in range(2001, 2021) for arm in ARMS)
    else:
        raise ValueError("expected_cases is defined only for development and evaluation")
    return [{"case_id": make_case_id(seed, arm, eta), "seed": seed, "arm": arm,
             "eta": float(eta)} for seed, arm, eta in triples]


def _json_safe(value: Any) -> Any:
    """Preserve nonfinite failure values explicitly without nonstandard JSON NaN."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite__": repr(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _encoded(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _sync_directory(path: Path) -> None:
    # POSIX directory fsync records entry changes; failures are not hidden.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_encoded(value))
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(path.parent)


def _read_json(path: Path) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise VerificationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise VerificationError(f"nonstandard nonfinite JSON number: {value}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=nonfinite)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON artifact {path.name}: {exc}") from exc


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise VerificationError("artifact path must be a nonempty POSIX relative path")
    rel = PurePosixPath(relative)
    if (rel.is_absolute() or rel.as_posix() != relative
            or any(part in ("..", ".", "") for part in relative.split("/"))):
        raise VerificationError(f"unsafe artifact path: {relative}")
    path = root.joinpath(*rel.parts)
    cursor = root
    for part in rel.parts:
        cursor /= part
        if cursor.is_symlink():
            raise VerificationError(f"symlink artifacts are forbidden: {relative}")
    if not path.is_file():
        raise VerificationError(f"missing artifact: {relative}")
    return path


def _hash_entry(root: Path, entry: Mapping[str, Any]) -> None:
    if not isinstance(entry, Mapping) or not {"path", "sha256"} <= set(entry):
        raise VerificationError("artifact entry requires path and sha256")
    claimed = entry.get("sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise VerificationError("invalid SHA256 identity")
    if sha256_file(_inside(root, entry.get("path"))) != claimed:
        raise VerificationError(f"artifact SHA256 mismatch: {entry.get('path')}")


def _validate_expected(stage: str, rows: Any) -> dict[str, dict[str, Any]]:
    if stage not in _SCHEMA:
        raise VerificationError(f"unknown evidence stage: {stage}")
    if not isinstance(rows, list) or not rows:
        raise VerificationError("expected cases must be a nonempty list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"case_id", "seed", "arm", "eta"}:
            raise VerificationError("each expected case needs exactly case_id, seed, arm, eta")
        try:
            case_id = make_case_id(row["seed"], row["arm"], row["eta"])
        except ValueError as exc:
            raise VerificationError(str(exc)) from exc
        if row["case_id"] != case_id or case_id in result:
            raise VerificationError("noncanonical or duplicate expected case identity")
        result[case_id] = row
    if stage == "development":
        prescribed = expected_cases("development")
    elif stage == "evaluation":
        diagnostic = {row["eta"] for row in rows if row["arm"] != "uniform-extra-replay"}
        extra = {row["eta"] for row in rows if row["arm"] == "uniform-extra-replay"}
        if len(diagnostic) != 1 or len(extra) != 1:
            raise VerificationError("evaluation must use one diagnostic and one extra-replay rate")
        try:
            prescribed = expected_cases("evaluation", diagnostic_eta=next(iter(diagnostic)),
                                        extra_eta=next(iter(extra)))
        except ValueError as exc:
            raise VerificationError(str(exc)) from exc
    else:
        prescribed = rows
    if {row["case_id"]: row for row in prescribed} != result:
        raise VerificationError(f"{stage} expected case grid differs from the protocol")
    return result


def _validate_schema(stage: str, schema: Any) -> None:
    if not isinstance(schema, dict):
        raise VerificationError("required_fields must map dotted paths to numeric shapes")
    for key, shape in _SCHEMA[stage].items():
        if schema.get(key) != shape:
            raise VerificationError(f"required stage schema missing or weakened: {key}")
    for key, shape in schema.items():
        if not isinstance(key, str) or not key or not isinstance(shape, list):
            raise VerificationError("invalid required field schema")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in shape):
            raise VerificationError("schema dimensions must be positive integers")


def _validate_provenance(root: Path, stage: str, provenance: Any) -> None:
    required = PROVENANCE_REQUIRED | ({"lock"} if stage == "evaluation" else set())
    if not isinstance(provenance, dict) or not required <= set(provenance):
        raise VerificationError("source, tests, protocol, data, indices and evaluation lock required")
    all_paths = set()
    for category, entries in provenance.items():
        if not isinstance(entries, list) or not entries:
            raise VerificationError(f"empty provenance category: {category}")
        for entry in entries:
            _hash_entry(root, entry)
            if not entry["path"].startswith(f"provenance/{category}/"):
                raise VerificationError("provenance must refer to its independent snapshot")
            if entry["path"] in all_paths:
                raise VerificationError("duplicate provenance artifact")
            all_paths.add(entry["path"])


def _validate_evaluation_lock(root: Path, provenance: Mapping[str, Any],
                              expected: Mapping[str, Any]) -> None:
    entries = provenance["lock"]
    if len(entries) != 1:
        raise VerificationError("evaluation requires exactly one frozen lock artifact")
    lock = _read_json(_inside(root, entries[0]["path"]))
    if (not isinstance(lock, dict) or lock.get("kind") != "evaluation-lock-v1"
            or lock.get("complete") is not True
            or lock.get("no_prior_test_selection") is not True):
        raise VerificationError("evaluation lock lacks complete provenance attestation")
    if (lock.get("seeds") != list(range(2001, 2021))
            or lock.get("arms") != list(ARMS)
            or lock.get("architecture") != [196, 32, 10]):
        raise VerificationError("evaluation lock seeds, arms, or architecture differ")
    rates = lock.get("rates")
    if not isinstance(rates, dict) or set(rates) != {"diagnostic_eta", "extra_eta"}:
        raise VerificationError("evaluation lock lacks the two selected rates")
    try:
        locked_cases = expected_cases("evaluation", **rates)
    except (TypeError, ValueError) as exc:
        raise VerificationError("invalid selected rates in evaluation lock") from exc
    if {row["case_id"]: row for row in locked_cases} != expected:
        raise VerificationError("case configurations differ from frozen evaluation rates")
    if lock.get("data_manifest_sha256") not in {entry["sha256"] for entry in provenance["data"]}:
        raise VerificationError("evaluation lock data hash differs from retained data provenance")


def _field(record: Mapping[str, Any], dotted: str) -> Any:
    value = record
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise VerificationError(f"missing required field: {dotted}")
        value = value[key]
    return value


def _finite_tree(value: Any, location: str = "metrics") -> None:
    if isinstance(value, Mapping):
        if "__nonfinite__" in value:
            raise VerificationError(f"nonfinite metric: {location}")
        for key, item in value.items():
            _finite_tree(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{location}[{index}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise VerificationError(f"nonfinite metric: {location}")


def _validate_record(root: Path, record: Any, expected: Mapping[str, Any],
                     schema: Mapping[str, Any], seen: set[str], stage: str = "software") -> None:
    if not isinstance(record, dict):
        raise VerificationError("case record must be an object")
    case_id = record.get("case_id")
    if not isinstance(case_id, str) or case_id not in expected:
        raise VerificationError(f"unexpected case identity: {case_id!r}")
    if case_id in seen:
        raise VerificationError(f"duplicate case identity: {case_id}")
    identity = {key: record.get(key) for key in ("case_id", "seed", "arm", "eta")}
    if identity != expected[case_id]:
        raise VerificationError(f"case configuration differs from manifest: {case_id}")
    # Avoid Python's True == 1 silently validating a boolean seed/eta.
    try:
        if make_case_id(record["seed"], record["arm"], record["eta"]) != case_id:
            raise VerificationError("case identity is noncanonical")
    except ValueError as exc:
        raise VerificationError(str(exc)) from exc
    if record.get("status") != "ok":
        raise VerificationError(f"case is not successful: {case_id}")
    checks = record.get("prediction_checks", {})
    if not isinstance(checks, dict) or any(value is not True for value in checks.values()):
        raise VerificationError(f"failed prediction check: {case_id}")
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 5:
        raise VerificationError("each run requires exactly five block checkpoints")
    if any(not isinstance(entry, dict) for entry in checkpoints):
        raise VerificationError("checkpoint entries must be objects")
    blocks = [entry.get("block") for entry in checkpoints]
    if any(isinstance(block, bool) or not isinstance(block, int) for block in blocks):
        raise VerificationError("checkpoint block IDs must be integers 0..4")
    if any(not isinstance(entry.get("path"), str) for entry in checkpoints):
        raise VerificationError("checkpoint paths must be strings")
    if sorted(blocks) != list(range(5)) or len({x.get("path") for x in checkpoints}) != 5:
        raise VerificationError("checkpoint blocks and paths must be distinct and cover 0..4")
    for entry in checkpoints:
        _hash_entry(root, entry)
    for dotted, shape in schema.items():
        value = _field(record, dotted)
        try:
            array = np.asarray(value)
            if array.shape != tuple(shape) or array.dtype.kind not in "iuf":
                raise VerificationError(f"wrong numeric shape for {dotted}; expected {shape}")
            if not np.isfinite(array).all():
                raise VerificationError(f"nonfinite values in {dotted}")
        except (TypeError, ValueError) as exc:
            raise VerificationError(f"invalid numeric values in {dotted}: {exc}") from exc
        if any(word in dotted.rsplit(".", 1)[-1] for word in ("matrix", "recall", "accuracy")):
            if np.any(array < 0) or np.any(array > 1):
                raise VerificationError(f"accuracy/recall must be fractions in [0,1]: {dotted}")
    _finite_tree(record.get("metrics", {}))
    if stage != "software":
        _validate_research_record(root, record)
    seen.add(case_id)


_RESEARCH_COUNTERS = {
    "committed_base_updates": 2500, "fresh_presentations": 40000,
    "current_repeat_presentations": 40, "historical_presentations": 24960,
    "unique_observations": 10000, "trainable_scalars": 6634,
    "trainable_raw_bytes": 26536, "reservoir_raw_bytes": 158600,
    "base_selected_items": 25000, "reservoir_counter": 10000,
    "candidate_draws": 124648,
}
_PAIR_HASHES = ("candidate_trace_sha256", "reservoir_trace_sha256",
                "quota_trace_sha256", "stream_sha256")
_CHECKPOINT_SHAPES = {"p0": (196, 32), "p1": (32,), "p2": (32, 10), "p3": (10,)}


def _ledger_integer(ledger: Mapping[str, Any], key: str) -> int:
    value = ledger.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VerificationError(f"research ledger needs a nonnegative integer: {key}")
    return value


def _validate_research_checkpoint(path: Path) -> None:
    try:
        # Check members and bounded uncompressed size before NumPy allocates.
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (len(members) != 4 or {entry.filename for entry in members}
                    != {key + ".npy" for key in _CHECKPOINT_SHAPES}
                    or sum(entry.file_size for entry in members) > 65536):
                raise VerificationError("research checkpoint must contain exactly four bounded arrays")
        with np.load(path, allow_pickle=False) as checkpoint:
            if len(checkpoint.files) != 4 or set(checkpoint.files) != set(_CHECKPOINT_SHAPES):
                raise VerificationError("research checkpoint must contain p0, p1, p2, p3 only")
            for key, shape in _CHECKPOINT_SHAPES.items():
                array = checkpoint[key]
                if array.shape != shape or array.dtype != np.dtype("float32"):
                    raise VerificationError(f"invalid model shape or float32 dtype: {key}")
                if not np.isfinite(array).all():
                    raise VerificationError(f"nonfinite research checkpoint array: {key}")
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise VerificationError(f"invalid research NPZ checkpoint {path.name}: {exc}") from exc


def _validate_research_record(root: Path, record: Mapping[str, Any]) -> None:
    ledger = record.get("ledger")
    if not isinstance(ledger, dict):
        raise VerificationError("research case requires its complete resource ledger")
    for key, expected in _RESEARCH_COUNTERS.items():
        if _ledger_integer(ledger, key) != expected:
            raise VerificationError(f"research ledger {key} differs from protocol: expected {expected}")
    extra = record["arm"] == "uniform-extra-replay"
    for key, expected in {"virtual_gradients": 0 if extra else 2500,
                          "candidate_forward_examples": 0 if extra else 249296,
                          "score_count": 0 if extra else 124648}.items():
        if _ledger_integer(ledger, key) != expected:
            raise VerificationError(f"research ledger {key} differs for arm {record['arm']}")
    allowed = _ledger_integer(ledger, "operations_allowed")
    used = _ledger_integer(ledger, "operations_used")
    residual = _ledger_integer(ledger, "residual_operations")
    extra_updates = _ledger_integer(ledger, "extra_updates")
    # Frozen logical-scalar-kernels-v1 graph, independent of learner helpers.
    # Base update: 713678; replay-only update: 278574.
    # Candidate counts: [16]*8 + [32]*4 + [48]*4 + [50]*2484.
    expected_budget = (6200562008, 6200428622, 133386, 15853) if extra else (6200562008, 6200562008, 0, 0)
    if (allowed, used, residual, extra_updates) != expected_budget:
        raise VerificationError("research arithmetic budget differs from frozen v1 graph")
    if allowed == 0 or used > allowed or residual != allowed - used:
        raise VerificationError("research operation cap or residual identity violated")
    if not extra and (residual != 0 or extra_updates != 0):
        raise VerificationError("diagnostic arms must spend the diagnostic budget without extra updates")
    _finite_tree(ledger, "ledger")
    for key in _PAIR_HASHES:
        value = record.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise VerificationError(f"research pairing requires a canonical SHA256: {key}")
    stream_path = _inside(root, f"{record['case_id']}/stream-indices.npy")
    if sha256_file(stream_path) != record["stream_sha256"]:
        raise VerificationError("stream source-index artifact hash differs from case record")
    try:
        ids = np.load(stream_path, allow_pickle=False)
        if (not isinstance(ids, np.ndarray) or ids.shape != (10000,)
                or ids.dtype.kind not in "iu" or len(np.unique(ids)) != 10000
                or np.any(ids < 0) or np.any(ids >= 60000)):
            raise VerificationError("research stream needs 10000 unique official-training source IDs")
    except (OSError, TypeError, ValueError) as exc:
        raise VerificationError(f"invalid research stream index artifact: {exc}") from exc
    audits = _read_cases(_inside(root, f"{record['case_id']}/mechanism-audit.jsonl"))
    if (len(audits) != 80 or any(not isinstance(row, dict) for row in audits)
            or [row.get("update") for row in audits] != list(range(525, 2501, 25))):
        raise VerificationError("research requires exactly 80 ordered audit opportunities at updates 525..2500")
    for row in audits:
        for key in ("selected_mean_harm", "eligible_mean_harm", "mir_highloss_overlap",
                    "selected_minus_shuffled_old_loss"):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise VerificationError(f"mechanism audit requires finite numeric result: {key}")
        if not 0 <= row["mir_highloss_overlap"] <= 1:
            raise VerificationError("mechanism audit overlap must be a fraction in [0,1]")
        if _ledger_integer(row, "audit_operations") == 0:
            raise VerificationError("mechanism audit must report positive operation cost")
        if "candidate_rank_vs_exhaustive_harm" not in row:
            raise VerificationError("mechanism audit requires rank correlation or explicit null for ties")
        rank = row["candidate_rank_vs_exhaustive_harm"]
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, (int, float))
                                 or not math.isfinite(rank) or not -1 <= rank <= 1):
            raise VerificationError("mechanism audit rank correlation must be finite in [-1,1] or null")
        _finite_tree(row, "mechanism_audit")
    for entry in record["checkpoints"]:
        _validate_research_checkpoint(_inside(root, entry["path"]))


def _validate_research_pairing(records: Sequence[Mapping[str, Any]]) -> None:
    by_seed: dict[int, dict[str, Any]] = {}
    for record in records:
        # Rates must not affect the named stream, admission, candidate or quota RNGs.
        identity = {key: record[key] for key in _PAIR_HASHES}
        identity["operations_allowed"] = record["ledger"]["operations_allowed"]
        previous = by_seed.setdefault(record["seed"], identity)
        if previous != identity:
            raise VerificationError(f"paired stream/reservoir/candidate/quota hash or budget mismatch at seed {record['seed']}")


def _read_cases(path: Path) -> list[Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VerificationError("missing raw case JSONL") from exc
    if not data or not data.endswith(b"\n"):
        raise VerificationError("empty or truncated raw case JSONL")
    rows = []
    for index, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            raise VerificationError(f"empty case line {index}")
        try:
            # The same duplicate-key and nonfinite checks as metadata.
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise VerificationError(f"duplicate JSON key on line {index}: {key}")
                    result[key] = value
                return result
            def reject_constant(value):
                raise VerificationError(f"nonfinite JSON on line {index}: {value}")
            rows.append(json.loads(line, object_pairs_hook=pairs, parse_constant=reject_constant))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VerificationError(f"malformed case JSONL line {index}") from exc
    return rows


def _artifact_inventory(root: Path) -> dict[str, str]:
    inventory = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise VerificationError(f"symlink artifacts are forbidden: {relative}")
        if path.is_file() and relative not in _RESERVED:
            inventory[relative] = sha256_file(path)
    return inventory


def verify_bundle(path: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Read-only independent verification; never infer rows or repair a bundle.

    ``require_complete=False`` is for the writer's pre-finalization gate. It still
    requires *all* successful cases, valid provenance and exact checkpoint coverage.
    """
    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise VerificationError("bundle must be an existing nonsymlink directory")
    if (root / "FAILED").exists() or (root / "INTERRUPTED").exists():
        raise VerificationError("bundle contains a failed/interrupted lifecycle marker")
    is_complete = (root / "COMPLETE").is_file()
    if require_complete and (not is_complete or (root / "RUNNING").exists()):
        raise VerificationError("bundle has no clean completion marker")
    manifest = _read_json(_inside(root, "manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise VerificationError("unsupported evidence manifest")
    stage = manifest.get("stage")
    expected = _validate_expected(stage, manifest.get("expected_cases"))
    schema = manifest.get("required_fields")
    _validate_schema(stage, schema)
    _validate_provenance(root, stage, manifest.get("provenance"))
    if stage == "evaluation":
        _validate_evaluation_lock(root, manifest["provenance"], expected)
    records = _read_cases(_inside(root, "cases.jsonl"))
    seen: set[str] = set()
    checkpoint_paths: set[str] = set()
    for record in records:
        _validate_record(root, record, expected, schema, seen, stage)
        paths = {entry["path"] for entry in record["checkpoints"]}
        if checkpoint_paths & paths:
            raise VerificationError("checkpoint artifacts cannot be shared between cases")
        checkpoint_paths.update(paths)
    if seen != set(expected):
        raise VerificationError(f"incomplete case coverage: {len(seen)}/{len(expected)}")
    if stage != "software":
        _validate_research_pairing(records)
    actual_inventory = _artifact_inventory(root)
    actual_checkpoints = checkpoint_paths | {
        name for name in actual_inventory
        if name.startswith("checkpoints/") or (
            not name.startswith("provenance/")
            and Path(name).name.startswith(("block-", "block_"))
            and Path(name).suffix in (".npz", ".bin"))
    }
    if actual_checkpoints != checkpoint_paths:
        raise VerificationError("checkpoint directory has missing or unlisted artifacts")
    inventory_file = root / "artifacts.json"
    if inventory_file.exists() or is_complete:
        declared_inventory = _read_json(_inside(root, "artifacts.json"))
        if declared_inventory != actual_inventory:
            raise VerificationError("artifact inventory differs: missing, modified, or unexpected files")
    if is_complete:
        completion = _read_json(_inside(root, "COMPLETE"))
        if not isinstance(completion, dict) or completion.get("artifacts_sha256") != sha256_file(inventory_file):
            raise VerificationError("completion artifact-inventory digest mismatch")
        if completion.get("case_count") != len(expected) or completion.get("stage") != stage:
            raise VerificationError("completion case count or stage differs from manifest")
    return {"stage": stage, "case_count": len(records), "complete": is_complete,
            "case_ids": sorted(seen), "manifest_sha256": sha256_file(root / "manifest.json"),
            "artifact_count": len(actual_inventory)}


class Bundle:
    """Exclusive fresh output writer. Use as a context manager around all work."""

    def __init__(self, path: str | Path, stage: str,
                 expected_cases: Sequence[Mapping[str, Any]],
                 provenance: Mapping[str, str | Path | Sequence[str | Path]], *,
                 required_fields: Mapping[str, Sequence[int]] | None = None):
        self.path = Path(path)
        expected_rows = _json_safe(list(expected_cases))
        self._expected = _validate_expected(stage, expected_rows)
        self._schema = dict(_SCHEMA[stage])
        for key, shape in (required_fields or {}).items():
            if key in self._schema and list(shape) != self._schema[key]:
                raise ValueError(f"cannot weaken mandatory stage schema: {key}")
            self._schema[key] = list(shape)
        _validate_schema(stage, self._schema)
        required = PROVENANCE_REQUIRED | ({"lock"} if stage == "evaluation" else set())
        if not required <= set(provenance):
            raise ValueError("incomplete provenance categories")
        self._seen: set[str] = set()
        self._closed = False
        self._completed = False
        self._failed = False
        self._stage = stage
        # mkdir(exist_ok=False) is the exclusive reservation. Never reuse output.
        self.path.mkdir(parents=False, exist_ok=False)
        _sync_directory(self.path.parent)
        try:
            _write_new(self.path / "RUNNING", {"stage": stage, "started_at": _now()})
            with (self.path / "cases.jsonl").open("xb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            _sync_directory(self.path)
            snapshots = self._snapshot_provenance(provenance)
            _write_new(self.path / "manifest.json", {
                "schema_version": 1, "stage": stage, "expected_cases": expected_rows,
                "required_fields": self._schema, "provenance": snapshots,
            })
        except BaseException as exc:
            self._mark_failure(exc)
            raise

    def _snapshot_provenance(self, provenance):
        snapshots = {}
        for category, inputs in sorted(provenance.items()):
            if not isinstance(category, str) or not category or not all(
                    char.isalnum() or char in "_-" for char in category):
                raise ValueError("provenance category must be a simple directory name")
            paths = [inputs] if isinstance(inputs, (str, Path)) else list(inputs)
            if not paths:
                raise ValueError(f"empty provenance category: {category}")
            entries = []
            for index, source in enumerate(paths):
                source = Path(source)
                if source.is_symlink() or not source.exists():
                    raise ValueError(f"missing or symlink provenance source: {source}")
                if source.is_dir() and self.path.resolve().is_relative_to(source.resolve()):
                    raise ValueError("output directory cannot be inside provenance source")
                sources = sorted(source.rglob("*")) if source.is_dir() else [source]
                for item in sources:
                    relative = item.relative_to(source) if source.is_dir() else Path(source.name)
                    if any(part in ("__pycache__", ".git", ".pytest_cache") for part in relative.parts):
                        continue
                    if item.is_symlink():
                        raise ValueError(f"symlink provenance source forbidden: {item}")
                    if not item.is_file():
                        continue
                    destination = self.path / "provenance" / category / str(index) / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with item.open("rb") as src, destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                        dst.flush()
                        os.fsync(dst.fileno())
                    _sync_directory(destination.parent)
                    entries.append({"path": destination.relative_to(self.path).as_posix(),
                                    "sha256": sha256_file(destination)})
            if not entries:
                raise ValueError(f"empty provenance snapshot: {category}")
            snapshots[category] = entries
        # Sync all newly created directory links before publishing the manifest.
        for directory in sorted((self.path / "provenance").rglob("*"), reverse=True):
            if directory.is_dir():
                _sync_directory(directory)
        _sync_directory(self.path / "provenance")
        _sync_directory(self.path)
        return snapshots

    def _mark_failure(self, exc: BaseException) -> None:
        self._failed = True
        marker = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAILED"
        if not (self.path / marker).exists():
            _write_new(self.path / marker, {"exception_type": type(exc).__name__,
                                            "message": str(exc), "recorded_at": _now()})

    def append_case(self, record: Mapping[str, Any]) -> None:
        """Persist complete raw evidence before checking success or predictions."""
        if self._closed or self._completed or self._failed:
            raise EvidenceError("bundle is closed, completed, or failed")
        try:
            serializable = _json_safe(record)
            encoded = _encoded(serializable)
        except (TypeError, ValueError) as exc:
            serializable = {"status": "serialization_failed", "raw_repr": repr(record),
                            "error": str(exc)}
            encoded = _encoded(serializable)
        try:
            with (self.path / "cases.jsonl").open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            _validate_record(self.path, serializable, self._expected, self._schema, self._seen, self._stage)
        except BaseException as exc:
            self._mark_failure(exc)
            raise

    def complete(self) -> dict[str, Any]:
        """Seal only after independent checks; no recovery, overwrite, or filtering."""
        if self._closed or self._completed or self._failed:
            raise EvidenceError("bundle is closed, completed, or failed")
        try:
            report = verify_bundle(self.path, require_complete=False)
            # fsync every artifact before committing the inventory and marker.
            inventory = _artifact_inventory(self.path)
            for relative in inventory:
                with _inside(self.path, relative).open("rb") as handle:
                    os.fsync(handle.fileno())
            _write_new(self.path / "artifacts.json", inventory)
            verify_bundle(self.path, require_complete=False)
            _write_new(self.path / "COMPLETE", {
                "stage": self._stage, "case_count": report["case_count"],
                "artifacts_sha256": sha256_file(self.path / "artifacts.json"),
            })
            (self.path / "RUNNING").unlink()
            _sync_directory(self.path)
            report = verify_bundle(self.path)
            self._completed = True
            return report
        except BaseException as exc:
            self._mark_failure(exc)
            raise

    def close(self) -> None:
        """Close without implying successful completion; unfinished RUNNING remains."""
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise EvidenceError("cannot enter a closed bundle")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc is not None:
                self._mark_failure(exc)
            elif not self._completed and not self._failed:
                self._mark_failure(EvidenceError("context exited before completion"))
        finally:
            self.close()
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
