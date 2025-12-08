import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path



def plot_intensity_cyto(csv_file: Path, output_dir: Path):
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
    sns.barplot(x='sample_name', y='cytoplasmic_intensity_mean', palette="deep", data=data)
     # add SD error bars 
    plt.errorbar(
        x=range(len(data)),
        y=data['cytoplasmic_intensity_mean'],
        yerr=data['cytoplasmic_intensity_std'],
        fmt='none',
        ecolor='black',
        capsize=5
    )
    plt.title('Cytoplasmic ntensity Distribution by Sample')
    plt.xlabel('Sample')
    plt.ylabel('Intensity')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plot_path = output_dir / "Cytoplasm_intensity_distribution_boxplot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def plot_intensity_nuclei(csv_file: Path, output_dir: Path):
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
    sns.barplot(x='sample_name', y='nuclear_intensity_mean', palette="deep", data=data)
     # add SD error bars 
    plt.errorbar(
        x=range(len(data)),
        y=data['nuclear_intensity_mean'],
        yerr=data['nuclear_intensity_std'],
        fmt='none',
        ecolor='black',
        capsize=5
    )
    plt.title('Nuclear intensity Distribution by Sample')
    plt.xlabel('Sample')
    plt.ylabel('Intensity')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plot_path = output_dir / "Nuclear_intensity_distribution_boxplot.png"
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
