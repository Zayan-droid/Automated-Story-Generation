"""ID and slug generators."""
import re
import uuid
from datetime import datetime


def new_project_id() -> str:
    """Compact, sortable project id: '20261215_a3f4'."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def slug(text: str, max_len: int = 40) -> str:
    """URL/file-safe slug."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] or "untitled"
