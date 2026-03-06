"""Shared CLI helpers: boolean parsing and YAML config loading."""

from typing import Any, Dict

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore


def parse_bool(value: Any) -> bool:
    """Coerce a CLI/config value to bool (handles string 'true'/'yes'/etc.)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return its contents as a dict."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configuration files. "
            "Install it with 'pip install PyYAML' or omit --config."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
