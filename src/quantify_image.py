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

# Directory for KD
kd_tiff = base_dir / "data/knockdown/tiff_h1792"
kd_mask = base_dir / "data/knockdown/masks_h1792"

# %% Quantify images and save csv file
output_csv = csv_dir / "intensity_measurements_h1792KD.csv"

print(f"Loading images from: {kd_tiff}")
print(f"Loading masks from: {kd_mask}")

data = extract_intensity_batch(image_dir=Path(kd_tiff), output_csv=output_csv, mask_dir=Path(kd_mask))

# %%
