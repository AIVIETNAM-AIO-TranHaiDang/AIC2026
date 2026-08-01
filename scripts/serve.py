"""Run the online retrieval service (API + operator UI).

Loads every persisted artifact (indexes, chronicle, keyframe metadata),
builds the Cortex against the configured endpoint, and serves the operator
console at http://{service.host}:{service.port}/.

Usage:
    python scripts/serve.py --config configs/t0.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.config import load_config  # noqa: E402
from aic.log import setup_logging  # noqa: E402
from aic.models_cache import apply_model_cache_env  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Profile config YAML.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(
        args.log_level,
        log_file=cfg.service.log_file,
        max_bytes=cfg.service.log_max_bytes,
        backup_count=cfg.service.log_backups,
    )
    apply_model_cache_env(cfg.paths.models_dir)

    import uvicorn

    from aic.service.app import create_app
    from aic.service.state import build_service_state

    app = create_app(build_service_state(cfg), cfg.service)
    uvicorn.run(app, host=cfg.service.host, port=cfg.service.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
