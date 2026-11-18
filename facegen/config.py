"""Settings: where projects and models live, and where external tools are."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def default_home() -> Path:
    env = os.environ.get("FACEGEN_HOME")
    if env:
        return Path(env)
    return Path.home() / "FaceGen"


@dataclass
class Config:
    home: str = field(default_factory=lambda: str(default_home()))
    blender: str | None = None
    meshroom: str | None = None
    mediapipe_model: str | None = None
    flame_model: str | None = None
    flame_embedding: str | None = None
    faces_folder: str | None = None  # ...\livecpk\Faces\Asset\model\character\face\real

    @property
    def home_path(self) -> Path:
        return Path(self.home)

    @property
    def projects_dir(self) -> Path:
        return self.home_path / "projects"

    @property
    def models_dir(self) -> Path:
        return self.home_path / "models"

    def mediapipe_model_path(self) -> Path:
        return Path(self.mediapipe_model) if self.mediapipe_model else self.models_dir / "face_landmarker.task"

    def flame_model_path(self) -> Path | None:
        if self.flame_model:
            return Path(self.flame_model)
        for name in ("generic_model.pkl", "flame2023.pkl", "flame2023_Open.pkl", "flame2020.pkl", "flame.npz"):
            p = self.models_dir / name
            if p.exists():
                return p
        return None

    def flame_embedding_path(self) -> Path | None:
        if self.flame_embedding:
            return Path(self.flame_embedding)
        p = self.models_dir / "mediapipe_landmark_embedding.npz"
        return p if p.exists() else None

    @classmethod
    def load(cls) -> "Config":
        home = default_home()
        path = home / "config.json"
        cfg = cls(home=str(home))
        if path.exists():
            data = json.loads(path.read_text())
            for k, v in data.items():
                if hasattr(cfg, k) and k != "home":
                    setattr(cfg, k, v)
        for key in ("blender", "meshroom", "flame_model", "faces_folder"):
            env = os.environ.get(f"FACEGEN_{key.upper()}")
            if env:
                setattr(cfg, key, env)
        return cfg

    def save(self) -> None:
        self.home_path.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("home")
        (self.home_path / "config.json").write_text(json.dumps(data, indent=2))
