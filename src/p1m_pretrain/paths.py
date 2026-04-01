from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _default_paths_file() -> Path:
    return Path(__file__).resolve().parents[2] / "paths.txt"


def _parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition("=")
        if not sep:
            raise ValueError(f"Invalid path config line in {path}: {line!r}")
        values[key.strip()] = value.strip()
    return values


def _resolve_value(config_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config_dir / candidate).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    paths_file: Path
    project_root: Path
    data_root: Path
    poly_any2any_root: Path
    transpolymer_repo: Path
    mmpolymer_repo: Path
    mmpolymer_data_root: Path
    bigsmiles_repo: Path
    wandb_root: Path

    @property
    def cache_dir(self) -> Path:
        return self.project_root / "cache"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / "checkpoints"

    @property
    def downstream_results_dir(self) -> Path:
        return self.project_root / "downstream_results"

    @property
    def openpolymer_dir(self) -> Path:
        return self.project_root / "openPolymer"

    @property
    def smi_ted_dir(self) -> Path:
        return self.checkpoints_dir / "smi_ted"

    @property
    def pi1m_csv(self) -> Path:
        return self.poly_any2any_root / "data" / "raw" / "pi1m" / "original" / "PI1M_v2.csv"

    @property
    def external_jsonl_specs(self) -> tuple[tuple[str, str, str], ...]:
        raw_root = self.poly_any2any_root / "data" / "raw"
        return (
            (str(raw_root / "openpoly" / "openpoly.jsonl"), "openpoly", "PSMILES"),
            (str(raw_root / "polymetrix" / "polymetrix.jsonl"), "polymetrix", "PSMILES"),
            (str(raw_root / "radonpy" / "radonpy.jsonl"), "radonpy", "smiles"),
            (str(raw_root / "bigsmiles_conversion" / "bigsmiles_conversion.jsonl"), "bigsmiles_conversion", "SMILES"),
        )


@lru_cache(maxsize=1)
def get_paths(paths_file: str | Path | None = None) -> ProjectPaths:
    config_path = Path(paths_file).expanduser().resolve() if paths_file else _default_paths_file()
    if not config_path.exists():
        raise FileNotFoundError(f"Path configuration file not found: {config_path}")

    raw = _parse_key_value_file(config_path)
    config_dir = config_path.parent
    required = (
        "project_root",
        "data_root",
        "poly_any2any_root",
        "transpolymer_repo",
        "mmpolymer_repo",
        "mmpolymer_data_root",
        "bigsmiles_repo",
        "wandb_root",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Missing required path config keys in {config_path}: {missing}")

    return ProjectPaths(
        paths_file=config_path,
        project_root=_resolve_value(config_dir, raw["project_root"]),
        data_root=_resolve_value(config_dir, raw["data_root"]),
        poly_any2any_root=_resolve_value(config_dir, raw["poly_any2any_root"]),
        transpolymer_repo=_resolve_value(config_dir, raw["transpolymer_repo"]),
        mmpolymer_repo=_resolve_value(config_dir, raw["mmpolymer_repo"]),
        mmpolymer_data_root=_resolve_value(config_dir, raw["mmpolymer_data_root"]),
        bigsmiles_repo=_resolve_value(config_dir, raw["bigsmiles_repo"]),
        wandb_root=_resolve_value(config_dir, raw["wandb_root"]),
    )
