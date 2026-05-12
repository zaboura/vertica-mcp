# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Abdelhak Zabour
#
# See LICENSE file for full license text.
import logging


def setup_logger(verbosity: int = 0) -> logging.Logger:
    """
    Configure and return the package logger.
    - 0 => CRITICAL
    - 1 => INFO
    - >=2 => DEBUG
    Always sets the logger level (even if handlers already exist).
    """
    logger = logging.getLogger("vertica-mcp")

    if verbosity <= 0:
        level = logging.CRITICAL
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    # Always apply the level (important across repeated calls in tests/CLI)
    logger.setLevel(level)

    # Ensure a stream handler exists with the same level
    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        # Align existing handler levels upward if needed
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level)

    # Keep root config consistent (safe in CLIs/tests)
    logging.basicConfig(level=level, force=True)
    return logger
