import os

def init_dist():
    """
    Minimal placeholder for distributed init.
    Defaults to single-process local run; can be extended later.
    """
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    backend = os.environ.get("DIST_BACKEND", "gloo")  # for CPU by default
    # Intentionally no torch.distributed init here (to avoid hard deps).
    return {
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": master_port,
        "BACKEND": backend,
        "WORLD_SIZE": int(os.environ.get("WORLD_SIZE", "1")),
        "RANK": int(os.environ.get("RANK", "0")),
    }