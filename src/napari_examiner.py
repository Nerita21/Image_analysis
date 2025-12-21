# %% Import necessary libraries
import napari
from utils import load_config
from package import view_tiff_directory, view_masks_napari
from pathlib import Path
# %% Load configuration
config, base_dir = load_config()

# %% Load paths
raw_image_dir = base_dir / config["data"]["raw_images_dir"]
raw_tiff_image_dir = base_dir / config["data"]["raw_tiff_images_dir"]
raw_masks_dir = base_dir / config["data"]["raw_masks_dir"]
sample_tiff_masks_dir = base_dir / config["data"]["sample_tiff_masks_dir"]
corrected_masks_dir = base_dir / config["data"]["corrected_masks_dir"]

# %% View masks in Napari (for manual correction)
view_masks_napari(raw_masks_dir, raw_tiff_image_dir)

# %% Paths for sample TIFF directories
ctrl_dir = sample_tiff_masks_dir/"sample_ctrl"
ctrl_treat_dir = sample_tiff_masks_dir/"sample_treated"
saber_dir = sample_tiff_masks_dir/"sample_saber_ctrl"
saber_treat_dir = sample_tiff_masks_dir/"sample_saber_treated"

# %% Paths for knockdown data
kd_hela_dir_ctrl = Path(base_dir / "data/knockdown/masks/Ctrl")
kd_h1792_dir_ctrl = Path(base_dir / "data/knockdown/masks_h1792/Ctrl")
hela_dir_ctrl = Path(base_dir / "data/knockdown/tiff/Ctrl")
h1792_dir_ctrl = Path(base_dir / "data/knockdown/tiff_h1792/Ctrl")

# %% View knockdown data in Napari
view_masks_napari(kd_hela_dir_ctrl, hela_dir_ctrl)
# %%
view_masks_napari(kd_h1792_dir_ctrl, h1792_dir_ctrl)
# %% View sample TIFF directory in Napari
view_tiff_directory(ctrl_dir)
# %%
view_tiff_directory(ctrl_treat_dir)

# %%
view_tiff_directory(saber_dir)
# %%
view_tiff_directory(saber_treat_dir)

# %%
view_tiff_directory(raw_tiff_image_dir)

# %%
view_masks_napari(raw_masks_dir, raw_tiff_image_dir)
# %% Open napari and import images manually if needed
viewer = napari.Viewer()

