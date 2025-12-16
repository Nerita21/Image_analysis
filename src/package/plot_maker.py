import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np



def plot_intensity_cyto(
    csv_file: Path,
    output_dir: Path,
    error_type: str = "sd"  # "sd" or "sem"
):
    """
    Generate and save cytoplasmic intensity plots grouped by sample.

    Each row in the CSV represents one image.
    Bars show the mean intensity per sample.
    Error bars show variation between images (SD or SEM).

    Args:
        csv_file (Path): Path to the CSV file containing intensity measurements.
        output_dir (Path): Directory to save the generated plots.
        error_type (str): "sd" for standard deviation, "sem" for standard error.
    """

    # Load data
    data = pd.read_csv(csv_file)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by sample (aggregate across images)
    grouped = (
        data
        .groupby('sample_name')
        .agg(
            mean_intensity=('cytoplasmic_intensity_mean', 'mean'),
            std_intensity=('cytoplasmic_intensity_mean', 'std'),
            n_images=('cytoplasmic_intensity_mean', 'count')
        )
        .reset_index()
    )

    # Choose error bars
    if error_type == "sd":
        yerr = grouped['std_intensity']
        error_label = "SD"
    elif error_type == "sem":
        yerr = grouped['std_intensity'] / np.sqrt(grouped['n_images'])
        error_label = "SEM"
    else:
        raise ValueError("error_type must be 'sd' or 'sem'")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", n_colors=len(grouped))

    sns.barplot(
        x='sample_name',
        y='mean_intensity',
        data=grouped,
        palette=palette
    )

    plt.errorbar(
        x=range(len(grouped)),
        y=grouped['mean_intensity'],
        yerr=yerr,
        fmt='none',
        ecolor='black',
        capsize=5
    )

    plt.title(f'Cytoplasmic Intensity by Sample (mean ± {error_label})')
    plt.xlabel('Sample')
    plt.ylabel('Mean Cytoplasmic Intensity')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = output_dir / f"Cytoplasm_intensity_by_sample_{error_label}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")

def plot_intensity_nuclei(
    csv_file: Path,
    output_dir: Path,
    error_type: str = "sd"  # "sd" or "sem"
):
    """
    Generate and save nuclei intensity plots grouped by sample.

    Each row in the CSV represents one image.
    Bars show the mean intensity per sample.
    Error bars show variation between images (SD or SEM).

    Args:
        csv_file (Path): Path to the CSV file containing intensity measurements.
        output_dir (Path): Directory to save the generated plots.
        error_type (str): "sd" for standard deviation, "sem" for standard error.
    """

    # Load data
    data = pd.read_csv(csv_file)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by sample (aggregate across images)
    grouped = (
        data
        .groupby('sample_name')
        .agg(
            mean_intensity=('nuclear_intensity_mean', 'mean'),
            std_intensity=('nuclear_intensity_mean', 'std'),
            n_images=('nuclear_intensity_mean', 'count')
        )
        .reset_index()
    )

    # Choose error bars
    if error_type == "sd":
        yerr = grouped['std_intensity']
        error_label = "SD"
    elif error_type == "sem":
        yerr = grouped['std_intensity'] / np.sqrt(grouped['n_images'])
        error_label = "SEM"
    else:
        raise ValueError("error_type must be 'sd' or 'sem'")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", n_colors=len(grouped))

    sns.barplot(
        x='sample_name',
        y='mean_intensity',
        data=grouped,
        palette=palette
    )

    plt.errorbar(
        x=range(len(grouped)),
        y=grouped['mean_intensity'],
        yerr=yerr,
        fmt='none',
        ecolor='black',
        capsize=5
    )

    plt.title(f'Nuclear Intensity by Sample (mean ± {error_label})')
    plt.xlabel('Sample')
    plt.ylabel('Mean Nuclear Intensity')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = output_dir / f"Nuclear_intensity_by_sample_{error_label}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved to {plot_path}")

def plot_intensity_ratio(csv_file: Path, output_dir: Path):
    """
    Generate and save intensity distribution plots from CSV data.

    Args:
        csv_file (Path): Path to the CSV file containing intensity measurements.
        output_dir (Path): Directory to save the generated plots.
    """
    # Load data
    data = pd.read_csv(csv_file)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot settings
    sns.set_style(style="whitegrid")
    sns.set_palette("Set2")


   # Generate pboxplot for intensity distributions between samples
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sample_name', y='nuclear_to_cytoplasmic_ratio', palette="deep", data=data)
    plt.title('Intensity Ratio between nuclei and cytoplasm by Sample')
    plt.xlabel('Sample')
    plt.ylabel('Ratio')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plot_path = output_dir / "intensity_ratio_barplot_std.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
