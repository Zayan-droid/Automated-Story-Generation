"""File / path helpers shared across phases."""
import json
import shutil
from pathlib import Path
from typing import Any, Union

from shared import constants


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_dir(project_id: str) -> Path:
    """Per-project output root. Reads OUTPUTS_DIR lazily so tests can monkeypatch it."""
    return ensure_dir(constants.OUTPUTS_DIR / project_id)


def asset_path(project_id: str, *parts: str) -> Path:
    """Build a path inside a project directory, creating parent dirs."""
    p = project_dir(project_id)
    for part in parts[:-1]:
        p = ensure_dir(p / part)
    return p / parts[-1] if parts else p


def write_json(path: Union[str, Path], data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def read_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_tree(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    src_p, dst_p = Path(src), Path(dst)
    if dst_p.exists():
        shutil.rmtree(dst_p)
    shutil.copytree(src_p, dst_p)
    return dst_p
