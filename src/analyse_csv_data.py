# %% This script analyses CSV data containing intensity measurements and creates plots.
# %% Import necessary libraries
import pandas as pd
from utils import load_config
from package.plot_maker import plot_intensity_cyto, plot_intensity_nuclei, plot_intensity_ratio

# %% Load configuration
config, base_dir = load_config()

# %% 
data_dir = base_dir / config["data"]["csv_dir"]
plot_dir = base_dir / config["data"]["plots_dir"]
# %% Load data (example for knockdown data)
data = pd.read_csv(data_dir / "intensity_measurements_h1792KD.csv")
output_dir = plot_dir / "plot_KD"

# %% Create plots
plot_intensity_cyto(data_dir / "intensity_measurements_h1792KD.csv", output_dir)
plot_intensity_nuclei(data_dir / "intensity_measurements_h1792KD.csv", output_dir)
plot_intensity_ratio(data_dir / "intensity_measurements_h1792KD.csv", output_dir)
