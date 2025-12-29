import logging
import os
import sys
from datetime import datetime

def setup_logging(log_name="logger"):
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    log_file_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        filename=log_file_path,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    # Add console handler to see logs in terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)
    
    return logging.getLogger(log_name)