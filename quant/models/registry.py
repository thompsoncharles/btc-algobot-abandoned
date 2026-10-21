"""
Model registry: versioned save/load with metadata, promotion, rollback.

Layout:
    quant_data/models/{name}/v{NNN}/model.joblib
    quant_data/models/{name}/v{NNN}/meta.json
    quant_data/models/{name}/ACTIVE          <- file containing active version

Retraining (quant.retrain) saves a new version, compares it against the
active one, and either promotes it or leaves the old version active.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
from loguru import logger

from quant import config


class ModelRegistry:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else config.MODEL_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _versions(self, name: str) -> list:
        return sorted(int(p.name[1:]) for p in self._model_dir(name).glob("v*") if p.is_dir())

    # ------------------------------------------------------------ save
    def save(self, name: str, model, metadata: dict) -> int:
        versions = self._versions(name)
        version = (versions[-1] + 1) if versions else 1
        vdir = self._model_dir(name) / f"v{version:03d}"
        vdir.mkdir(parents=True)
        joblib.dump(model, vdir / "model.joblib")
        meta = dict(metadata)
        meta.update({
            "name": name, "version": version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        })
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info("saved model {} v{:03d}", name, version)
        return version

    # ------------------------------------------------------------ load
    def load(self, name: str, version: Optional[int] = None) -> Tuple[object, dict]:
        if version is None:
            version = self.active_version(name)
            if version is None:
                versions = self._versions(name)
                if not versions:
                    raise FileNotFoundError(f"no saved versions of model '{name}'")
                version = versions[-1]
        vdir = self._model_dir(name) / f"v{version:03d}"
        model = joblib.load(vdir / "model.joblib")
        meta = json.loads((vdir / "meta.json").read_text())
        return model, meta

    # ------------------------------------------------- promote/rollback
    def active_version(self, name: str) -> Optional[int]:
        marker = self._model_dir(name) / "ACTIVE"
        if marker.exists():
            return int(marker.read_text().strip())
        return None

    def promote(self, name: str, version: int) -> None:
        (self._model_dir(name) / "ACTIVE").write_text(str(version))
        logger.info("model {} v{:03d} promoted to ACTIVE", name, version)

    def rollback(self, name: str) -> Optional[int]:
        """Reactivate the version prior to the current active one."""
        current = self.active_version(name)
        versions = self._versions(name)
        candidates = [v for v in versions if current is None or v < current]
        if not candidates:
            logger.warning("no earlier version of {} to roll back to", name)
            return None
        self.promote(name, candidates[-1])
        return candidates[-1]
