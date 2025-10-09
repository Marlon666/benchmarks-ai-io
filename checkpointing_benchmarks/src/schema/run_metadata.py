import datetime as _dt
import platform as _platform
import socket as _socket
from typing import Any, Dict


def default_metadata(run_name: str, params: Dict[str, Any],
                     summary: Dict[str, Any]) -> Dict[str, Any]:
    now = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return {
        "run_name": run_name,
        "timestamp_utc": now,
        "host": {
            "hostname": _socket.gethostname(),
            "platform": _platform.platform(),
            "python_version": _platform.python_version(),
        },
        "parameters": dict(params),
        "summary": dict(summary),
    }

