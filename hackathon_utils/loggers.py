import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger():
    # Define the log file path
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True) # Create logs directory if it doesn't exist
    log_file = os.path.join(log_directory, f"training.log")
    
    # 1. Get a logger instance
    logger = logging.getLogger(__name__) # __name__ gives a hierarchical logger name
    logger.setLevel(logging.DEBUG) # Set the logging level for the logger
    
    # 2. Create a file handler
    # RotatingFileHandler: rotates logs when the file reaches a certain size
    # maxBytes: maximum size of the file before rotation (e.g., 1MB = 1024*1024 bytes)
    # backupCount: number of backup log files to keep
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 5, backupCount=5) # 5MB per file, keep 5 backups
    file_handler.setLevel(logging.INFO) # Set the logging level for this handler
    
    # 3. Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 4. Set the formatter for the file handler
    file_handler.setFormatter(formatter)
    
    # 5. Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger