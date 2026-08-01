"""Logging setup shared by CLI scripts and library code.

Library modules obtain loggers with ``logging.getLogger(__name__)`` and never
configure handlers themselves; only entry points call :func:`setup_logging`.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# Third-party loggers that emit one INFO line per HTTP request; at INFO they
# bury the pipeline's own progress output (httpx logs every completion call).
_CHATTY_LOGGERS = ("httpx", "httpcore")


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    quiet_loggers: tuple[str, ...] = _CHATTY_LOGGERS,
) -> None:
    """Configure the root logger once, for scripts and tests.

    ``log_file`` adds a rotating file handler alongside stderr (Phase 9
    hardening: a long round must never fill the disk with logs). Rotation
    limits come from ``service.log_max_bytes`` / ``service.log_backups``.
    ``quiet_loggers`` are capped at WARNING so per-request client noise
    never drowns the job progress bars; pass ``()`` to keep them verbose
    (e.g. when debugging endpoint traffic).
    """
    logging.basicConfig(level=level.upper(), format=_FORMAT, datefmt=_DATE_FORMAT)
    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_FILE_DATE_FORMAT))
        logging.getLogger().addHandler(handler)
