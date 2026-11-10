import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Project root directory for log storage
LOGS_DIR = Path(__file__).resolve().parent.parent / "data" / "output" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = LOGS_DIR / "app.log"

# Standard production log format
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = DEFAULT_LOG_FILE,
) -> logging.Logger:
    """Configure structured root logger with console and optional file handler."""
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    root_logger = logging.getLogger("clinical_trial_rag_copilot")
    
    # Avoid attaching duplicate handlers if already initialized
    if not root_logger.handlers:
        root_logger.setLevel(log_level)

        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

        # 1. Console stream handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 2. File handler (persistent logging in data/output/logs/app.log)
        if log_file:
            try:
                file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                # If file cannot be written, fallback gracefully to console-only
                root_logger.warning("Could not initialize file handler at %s: %s", log_file, e)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve namespaced child logger configured with standard project settings."""
    # Ensure root logger initialized
    setup_logging()
    
    # Return child logger scoped under our package namespace
    if not name.startswith("clinical_trial_rag_copilot"):
        name = f"clinical_trial_rag_copilot.{name}"
    return logging.getLogger(name)
