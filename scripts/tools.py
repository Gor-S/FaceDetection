import yaml
import os
import shutil
import gc
import torch

def remove_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)

# Load config file
def load_config(config_path):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("❌ Error: Config file not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        exit(1)

def clear_memory():
    """ Free memory and clear GPU cache if available. """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠️ GPU memory cleanup error: {e}")
