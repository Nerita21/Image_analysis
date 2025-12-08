# %% Import necessary libraries
import pandas as pd
from utils import load_config
from package.plot_maker import plot_intensity_cyto, plot_intensity_nuclei, plot_intensity_ratio, plot_intensity_ratio_std

# %% Load configuration
config, base_dir = load_config()

# %% 
data_dir = base_dir / config["data"]["csv_dir"]
data = pd.read_csv(data_dir / "intensity_measurements.csv")
output_dir = data_dir / "plots"

# %% Create plots
plot_intensity_cyto(data_dir / "intensity_measurements.csv", output_dir)
plot_intensity_nuclei(data_dir / "intensity_measurements.csv", output_dir)
plot_intensity_ratio(data_dir / "intensity_measurements.csv", output_dir)

# %%
plot_intensity_ratio_std(data_dir / "intensity_measurements.csv", output_dir)
# %%
