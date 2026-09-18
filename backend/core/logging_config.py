"""
InsureAI — Structured Logging Configuration
=============================================
Requirements 407-408:
  - Structured logging across backend
  - Replaces arbitrary print() statements
  - Windows encoding safe
"""

import sys
import logging


def setup_logging(level: str = "INFO") -> None:
    """Configures application-wide logging with uniform timestamped format."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on re-import
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers[0].setFormatter(formatter)
