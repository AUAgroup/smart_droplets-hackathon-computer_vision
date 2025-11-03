import os
import os, sys, logging, logging.config
from datetime import datetime

def set_log():
    # 1) Path for the common log file
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, f"session-{datetime.now():%Y%m%d-%H%M%S}.log")
    # 3) Central config: everything routes to 'file', optionally also to stdout
    LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,  # keep 3rd-party loggers, just re-route them
        "formatters": {
            "std": {
                "format": "%(asctime)s %(levelname).1s [%(name)s:%(lineno)d] %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": LOG_PATH,
                "maxBytes": 5_000_000,
                "backupCount": 3,
                "encoding": "utf-8",
                "formatter": "std",
            },
        },
        "root": {  # the root logger – all module logs propagate here by default
            "level": "INFO",
            "handlers": ["file"],  # drop "stdout" if you want file-only
        },
    }

    return LOG_CONFIG