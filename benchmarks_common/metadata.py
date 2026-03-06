"""Consistent run-metadata schema shared across all benchmark modules."""

import datetime as _dt
import platform as _platform
import socket as _socket
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class RunMetadata:
    run_name: str
    benchmark: str  # e.g. "listing", "checkpointing", "serving", "dataloader"
    timestamp_utc: str = ""
    host: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp_utc:
            self.timestamp_utc = (
                _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
        if not self.host:
            self.host = {
                "hostname": _socket.gethostname(),
                "platform": _platform.platform(),
                "python_version": _platform.python_version(),
            }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_metadata(
    run_name: str,
    benchmark: str,
    parameters: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Convenience builder that returns a plain dict ready for YAML output."""
    return RunMetadata(
        run_name=run_name,
        benchmark=benchmark,
        parameters=parameters,
        summary=summary,
    ).to_dict()
