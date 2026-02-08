import logging
import os
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(log_name="logger"):
    """
    Setup centralized logging for the project.
    
    Args:
        log_name (str): Name of the logger (helps identify which script is logging)
    
    Returns:
        logging.Logger: Configured logger instance
    
    This creates:
    - A single 'logs' folder at the PROJECT_ROOT level
    - Separate log files for each script execution with timestamp and script name
    - Console output for real-time monitoring
    """
    # Find project root (2 levels up from src/utility/)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    
    # Create centralized logs directory at project root
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log file with script name and timestamp
    timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    log_file = f"{log_name}_{timestamp}.log"
    log_file_path = log_dir / log_file
    
    # Get or create logger
    logger = logging.getLogger(log_name)
    
    # Prevent duplicate handlers if logger already exists
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate logs from parent loggers
    
    # Formatter for detailed file logs
    file_formatter = logging.Formatter(
        "[ %(asctime)s ] %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    
    # Formatter for cleaner console output
    console_formatter = logging.Formatter(
        "%(levelname)s | %(name)s | %(message)s"
    )
    
    # File Handler - detailed logs
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console Handler - cleaner output for terminal visibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger '{log_name}' initialized. Log file: {log_file_path}")
    
    return logger