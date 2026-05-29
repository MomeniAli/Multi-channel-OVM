"""
Helpers for loading, overriding, and snapshotting ONN training configuration.
"""

from __future__ import annotations

import copy
from pathlib import Path
import secrets
from typing import Any, Dict, Optional, Tuple

import yaml

_DEFAULT_CFG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        keys = key.split(".")
        cursor = cfg
        for sub_key in keys[:-1]:
            if sub_key not in cursor or not isinstance(cursor[sub_key], dict):
                cursor[sub_key] = {}
            cursor = cursor[sub_key]
        cursor[keys[-1]] = value
    return cfg


def load_training_config(
    config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Path]:
    """
    Load training configuration from YAML and apply dotted-key overrides.
    """
    cfg_path = Path(config_path) if config_path else _DEFAULT_CFG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    return cfg, cfg_path


def resolve_onn_seed(cfg: Dict[str, Any], default: int = 1337) -> int:
    """
    Resolve (and persist) the ONN RNG seed for this run.

    Accepted values for `cfg['onn']['seed']`:
    - int >= 0: fixed seed (reproducible).
    - null/None: pick a fresh random seed each run.
    - int < 0: pick a fresh random seed each run.
    - "random"/"auto"/"none" (case-insensitive): pick a fresh random seed each run.

    The resolved integer seed is written back into `cfg['onn']['seed']` so that
    `write_config_snapshot(...)` captures the exact seed used.
    """
    onn_cfg = cfg.get("onn")
    if not isinstance(onn_cfg, dict):
        onn_cfg = {}
        cfg["onn"] = onn_cfg

    seed_raw = onn_cfg.get("seed", default)

    if seed_raw is None:
        seed = int(secrets.randbelow(2**31 - 1))
    elif isinstance(seed_raw, str):
        token = seed_raw.strip().lower()
        if token in {"random", "rand", "auto", "none", "null", "~"}:
            seed = int(secrets.randbelow(2**31 - 1))
        else:
            try:
                seed = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid onn.seed={seed_raw!r}. Use an int, null, -1, or 'random'."
                ) from exc
    else:
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid onn.seed={seed_raw!r}. Use an int, null, -1, or 'random'."
            ) from exc
        if seed < 0:
            seed = int(secrets.randbelow(2**31 - 1))

    onn_cfg["seed"] = seed
    return seed


def write_config_snapshot(cfg: Dict[str, Any], run_dir: Path, filename: str = "cfg_latest.yaml") -> Path:
    """
    Persist the effective configuration next to the run artifacts for reproducibility.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / filename
    with open(output_path, "w") as handle:
        yaml.safe_dump(cfg, handle)
    return output_path
