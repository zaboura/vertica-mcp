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


def setup_logger(verbose: int) -> logging.Logger:
    logger = logging.getLogger("vertica-mcp")
    logger.propagate = False
    level = logging.CRITICAL
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        if verbose == 0:
            handler.setLevel(logging.CRITICAL)
            logger.setLevel(logging.CRITICAL)
        elif verbose == 1:
            handler.setLevel(logging.INFO)
            logger.setLevel(logging.INFO)
            level = logging.INFO
        else:
            handler.setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
            level = logging.DEBUG
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logging.basicConfig(level=level, force=True)
    return logger
