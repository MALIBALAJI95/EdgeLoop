"""Configuration manager for EdgeLoop."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class Config:
    """System configuration loader with YAML and environment variable support."""

    def __init__(self, config_path: Optional[str] = None):
        self.root_dir = Path(__file__).resolve().parents[3]
        if config_path is None:
            config_path = os.getenv("EDGELOOP_CONFIG_PATH", str(self.root_dir / "configs" / "config.yaml"))
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = self.root_dir / self.config_path

        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieve nested config value by dot notation, e.g., 'model.num_classes'."""
        keys = key_path.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def classes(self) -> List[str]:
        return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    @property
    def database_path(self) -> Path:
        env_db = os.getenv("EDGELOOP_DB_PATH")
        if env_db:
            p = Path(env_db)
            return p if p.is_absolute() else self.root_dir / p
        rel_p = self.get("paths.database_path", "artifacts/registry/edgeloop.db")
        return self.root_dir / rel_p

    @property
    def baseline_checkpoint_path(self) -> Path:
        rel_p = self.get("model.baseline_checkpoint", "research/edgeloop/mobilenetv2_baseline.pth")
        return self.root_dir / rel_p

    @property
    def champion_checkpoint_path(self) -> Path:
        rel_p = self.get("model.champion_checkpoint", "artifacts/champion/mobilenetv2_champion.pth")
        return self.root_dir / rel_p

    @property
    def champion_dir(self) -> Path:
        rel_p = self.get("paths.champion_dir", "artifacts/champion")
        return self.root_dir / rel_p

    @property
    def candidates_dir(self) -> Path:
        rel_p = self.get("paths.candidates_dir", "artifacts/candidates")
        return self.root_dir / rel_p


# Global singleton
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = Config(config_path)
    return _config_instance
