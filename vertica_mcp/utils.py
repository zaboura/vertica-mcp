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
