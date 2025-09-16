from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time, socket, platform

@dataclass
class RunMetadata:
    run_name: str
    timestamp: float
    mode: str                  # "train" or "infer"
    steps: int
    env: Dict[str, Any]
    summary: Dict[str, Any]    # e.g., mean throughput, P95 latency, etc.

def default_metadata(run_name: str, mode: str, steps: int, summary=None) -> RunMetadata:
    env = {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    return RunMetadata(
        run_name=run_name,
        timestamp=time.time(),
        mode=mode,
        steps=steps,
        env=env,
        summary=summary or {},
    )