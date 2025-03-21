import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(log_path="logs/application.log", max_log_size=10 * 1024 * 1024, backup_count=5):
    """
    Set up the logger with a rotating file handler.
    :param log_path: Path to the log file (can be customized per module).
    :param max_log_size: Max size of log file before rotating.
    :param backup_count: Number of backup log files to keep.
    :return: Logger instance.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger instance
    logger = logging.getLogger("ApplicationLogger")
    logger.setLevel(logging.INFO)

    # Create a rotating file handler
    handler = RotatingFileHandler(log_path, maxBytes=max_log_size, backupCount=backup_count)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger
