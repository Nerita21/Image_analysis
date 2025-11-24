from pathlib import Path
import yaml

def load_config():
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / 'config_ex.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config, base_dir
