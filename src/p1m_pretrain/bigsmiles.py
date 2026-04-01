from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pandas as pd


BIGSMILES_REPO = Path("/mnt/data/tmp/bigsmiles_repo")
BIGSMILES_MODULE_PATH = BIGSMILES_REPO / "BigSMILES_homopolymer" / "BigSMILES_homopolymer.py"

_BIGSMILES_MODULE: ModuleType | None = None
_BIGSMILES_CONVERTER = None


def _load_bigsmiles_module() -> ModuleType:
    global _BIGSMILES_MODULE
    if _BIGSMILES_MODULE is not None:
        return _BIGSMILES_MODULE
    if not BIGSMILES_MODULE_PATH.exists():
        raise FileNotFoundError(f"BigSMILES converter module not found: {BIGSMILES_MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("bigsmiles_homopolymer", BIGSMILES_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import BigSMILES converter from {BIGSMILES_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BIGSMILES_MODULE = module
    return module


def _get_converter():
    global _BIGSMILES_CONVERTER
    if _BIGSMILES_CONVERTER is None:
        module = _load_bigsmiles_module()
        _BIGSMILES_CONVERTER = module.SMILES2BigSMILES()
    return _BIGSMILES_CONVERTER


def convert_psmiles_to_bigsmiles(psmiles: str) -> str | None:
    try:
        converted = _get_converter().Converting_single(SMILES=str(psmiles))
    except Exception:
        return None
    if converted in (0, None, ""):
        return None
    return str(converted)


def _convert_chunk(psmiles_values: list[str]) -> list[str | None]:
    converter = _get_converter()
    outputs: list[str | None] = []
    for psmiles in psmiles_values:
        try:
            converted = converter.Converting_single(SMILES=str(psmiles))
        except Exception:
            converted = None
        if converted in (0, None, ""):
            outputs.append(None)
        else:
            outputs.append(str(converted))
    return outputs


def _chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def augment_parquet_with_bigsmiles(
    input_path: Path,
    output_path: Path,
    *,
    psmiles_column: str = "psmiles",
    preprocess_workers: int = 8,
    drop_missing: bool = False,
) -> Path:
    if output_path.exists():
        print(f"Using existing BigSMILES cache: {output_path}")
        return output_path

    df = pd.read_parquet(input_path)
    if psmiles_column not in df.columns:
        raise KeyError(f"Column {psmiles_column!r} not found in {input_path}")

    psmiles_values = df[psmiles_column].astype(str).tolist()
    workers = max(1, min(preprocess_workers, os.cpu_count() or 1))
    print(
        f"Building BigSMILES cache from {input_path} -> {output_path} "
        f"for {len(psmiles_values)} records with {workers} worker(s)"
    )
    if workers == 1:
        bigsmiles_values = _convert_chunk(psmiles_values)
    else:
        chunk_size = max(256, len(psmiles_values) // (workers * 8) or 256)
        chunks = _chunked(psmiles_values, chunk_size)
        bigsmiles_values = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for converted_chunk in executor.map(_convert_chunk, chunks, chunksize=1):
                bigsmiles_values.extend(converted_chunk)

    df["bigsmiles"] = bigsmiles_values
    success_count = int(df["bigsmiles"].notna().sum())
    if drop_missing:
        df = df.dropna(subset=["bigsmiles"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(
        f"Saved BigSMILES cache to {output_path} "
        f"({success_count}/{len(psmiles_values)} converted, drop_missing={int(drop_missing)})"
    )
    return output_path
