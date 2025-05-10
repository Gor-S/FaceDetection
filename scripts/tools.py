import yaml
import os
import shutil
import gc
import torch
from datetime import datetime
from . import logger

# Initialize logger
logger = logger.get_logger(module_name="tools")

def remove_folder(folder):
    if os.path.exists(folder):
        logger.info(f"Removing folder: {folder}")
        shutil.rmtree(folder)
        logger.debug(f"Folder {folder} removed successfully")
    else:
        logger.debug(f"Folder {folder} does not exist, nothing to remove")

# Load config file
def load_config(config_path):
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.debug(f"Configuration loaded successfully: {config}")
            return config
    except FileNotFoundError:
        error_msg = "Config file not found."
        logger.critical(f"Error: {error_msg}")
        exit(1)
    except yaml.YAMLError as e:
        error_msg = f"YAML parsing error: {e}"
        logger.critical(error_msg)
        exit(1)

def clear_memory():
    """ Free memory and clear GPU cache if available. """
    logger.debug("Clearing system memory")
    gc.collect()
    if torch.cuda.is_available():
        try:
            logger.debug("Clearing GPU memory cache")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            error_msg = f"GPU memory cleanup error: {e}"
            logger.warning(error_msg)


def generate_folder_name():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("output/best_faces", timestamp)
    return save_dir