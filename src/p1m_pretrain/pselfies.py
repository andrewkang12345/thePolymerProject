from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

from rdkit import Chem, RDLogger
import selfies as sf


RDLogger.DisableLog("rdApp.*")


def canonical_proxy_smiles_from_psmiles(psmiles: str) -> str | None:
    if not psmiles:
        return None
    try:
        msmiles = psmiles.replace("[*]", "[At]").replace("*", "[At]")
        mol = Chem.MolFromSmiles(msmiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def proxy_pselfies_from_psmiles(psmiles: str) -> str | None:
    try:
        canonical = canonical_proxy_smiles_from_psmiles(psmiles)
        if canonical is None:
            return None
        return sf.encoder(canonical)
    except Exception:
        return None


def randomize_psmiles(psmiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(psmiles)
        if mol is None:
            return psmiles
        randomized = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        return randomized or psmiles
    except Exception:
        return psmiles


def tokenize_representation(text: str, rep_type: str) -> list[str]:
    if rep_type == "pselfies":
        return list(sf.split_selfies(text))
    if rep_type in {"psmiles", "bigsmiles"}:
        return list(text)
    raise ValueError(f"Unsupported representation type: {rep_type}")


@dataclass
class RepresentationVocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    PAD: str = "<pad>"
    BOS: str = "<bos>"
    EOS: str = "<eos>"
    UNK: str = "<unk>"
    MASK: str = "<mask>"

    @classmethod
    def build(
        cls,
        psmiles_values: Iterable[str],
        pselfies_values: Iterable[str],
        bigsmiles_values: Iterable[str] | None = None,
    ) -> "RepresentationVocab":
        tokens = {cls.PAD, cls.BOS, cls.EOS, cls.UNK, cls.MASK}
        for value in psmiles_values:
            tokens.update(tokenize_representation(value, "psmiles"))
        for value in pselfies_values:
            tokens.update(tokenize_representation(value, "pselfies"))
        if bigsmiles_values is not None:
            for value in bigsmiles_values:
                tokens.update(tokenize_representation(value, "bigsmiles"))
        ordered = [cls.PAD, cls.BOS, cls.EOS, cls.UNK, cls.MASK] + sorted(
            tokens - {cls.PAD, cls.BOS, cls.EOS, cls.UNK, cls.MASK}
        )
        return cls(token_to_id={token: idx for idx, token in enumerate(ordered)}, id_to_token=ordered)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.UNK]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[self.MASK]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, rep_type: str, max_length: int) -> list[int]:
        tokens = tokenize_representation(text, rep_type)
        ids = [self.bos_id]
        ids.extend(self.token_to_id.get(token, self.unk_id) for token in tokens[: max_length - 2])
        ids.append(self.eos_id)
        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids[:max_length]

    def decode(self, ids: Iterable[int]) -> list[str]:
        tokens = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if token in {self.PAD, self.BOS}:
                continue
            if token == self.EOS:
                break
            tokens.append(token)
        return tokens

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"id_to_token": self.id_to_token}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "RepresentationVocab":
        payload = json.loads(path.read_text())
        tokens = payload["id_to_token"]
        if cls.MASK not in tokens:
            tokens = tokens[:4] + [cls.MASK] + tokens[4:]
        return cls(token_to_id={token: idx for idx, token in enumerate(tokens)}, id_to_token=tokens)
