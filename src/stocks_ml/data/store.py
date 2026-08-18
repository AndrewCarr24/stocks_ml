from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

VALID_NAMES = {"prices", "membership", "fred", "edgar", "panel", "removals", "form4", "shortint", "sec8k",
    "sharadar_prices", "sharadar_tickers", "sharadar_sp500",
}


class DataStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest: dict | None = None

    def _path(self, name: str) -> Path:
        if name not in VALID_NAMES:
            raise ValueError(f"unknown dataset {name!r}; valid: {sorted(VALID_NAMES)}")
        return self.root / f"{name}.parquet"

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def read(self, name: str) -> pd.DataFrame:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist; run `stocks-ml ingest` first")
        return pd.read_parquet(path)

    def write(self, name: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(name), index=False)

    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            mpath = self.root / "manifest.json"
            self._manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        return self._manifest

    def set_manifest(self, key: str, value) -> None:
        self.manifest[key] = value
        (self.root / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))
