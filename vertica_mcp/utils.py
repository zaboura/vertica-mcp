# Copyright 2025 Abdelhak Zabour
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
