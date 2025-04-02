import logging
import os
import sys
from datetime import datetime
class Logger:
    """Custom logger that redirects all output to files instead of console"""
    
    def __init__(self, log_dir="logs", module_name=None):
        """
        Initialize logger with log directory and optional module name
        
        Args:
            log_dir (str): Directory to store log files
            module_name (str, optional): Name of the module for log filename
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create module-specific filename if provided, otherwise use generic name
        if module_name:
            log_filename = f"{module_name}_{timestamp}.log"
        else:
            log_filename = f"main_{timestamp}.log"
        
        self.log_path = os.path.join(log_dir, log_filename)
        
        self.logger = logging.getLogger(module_name or "main")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
    
    def debug(self, message):
        self.logger.debug(message)
        
    def info(self, message):
        self.logger.info(message)
        
    def warning(self, message):
        self.logger.warning(message)
        
    def error(self, message):
        self.logger.error(message)
        
    def critical(self, message):
        self.logger.critical(message)

def get_logger(module_name=None):
    """
    Get a logger instance for the specified module
    
    Args:
        module_name (str, optional): Name of the module
        
    Returns:
        Logger: Logger instance
    """
    return Logger(module_name=module_name)

def redirect_stdout_stderr(log_dir="logs"):
    """
    Redirect stdout and stderr to log files
    
    Args:
        log_dir (str): Directory to store log files
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Redirect stdout
    stdout_log = os.path.join(log_dir, f"stdout_{timestamp}.log")
    sys.stdout = open(stdout_log, 'w')
    
    # Redirect stderr
    stderr_log = os.path.join(log_dir, f"stderr_{timestamp}.log")
    sys.stderr = open(stderr_log, 'w')