# %% Import necessary libraries
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils import load_config
from package.intensity_measurer import extract_intensity_batch

# %% Load configuration
config, base_dir = load_config()

# %% Set up paths for intensity images and masks
csv_dir = base_dir / config["data"]["csv_dir"]

# Intensity images directory (ch1, ch2 files) - data/images/tiff with sample subfolders
image_dir = base_dir / config["data"]["tiff_images_dir"]

# Masks directory (separate folder with *_mask files) - data/images/masks with sample subfolders
mask_dir = base_dir / "data/images/masks"

# %% Quantify images and save csv file
output_csv = csv_dir / "intensity_measurements.csv"

print(f"Loading images from: {image_dir}")
print(f"Loading masks from: {mask_dir}")

data = extract_intensity_batch(image_dir, output_csv, mask_dir=mask_dir)
