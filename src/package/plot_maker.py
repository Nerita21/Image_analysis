import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from statannotations.Annotator import Annotator


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

def plot_intensity_ratio_stat(csv_file: Path, output_dir: Path):
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

    # Prepare pairwise comparisons
    samples = data["sample_name"].unique()
    pairs = [(samples[i], samples[j])
             for i in range(len(samples))
             for j in range(i + 1, len(samples))]


   # Generate pboxplot for intensity distributions between samples
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='sample_name', y='nuclear_to_cytoplasmic_ratio', palette="deep", data=data, ci="sd")

    max_val = data["nuclear_to_cytoplasmic_ratio"].max()
    ax.set_ylim(0, max_val * 1.5)  # Add some space for annotations

    # Add statistical annotations
    annotator = Annotator(
        ax,
        pairs,
        data=data,
        x="sample_name",
        y="nuclear_to_cytoplasmic_ratio"
    )
    annotator.configure(
        test="t-test_ind",        # choose your test
        text_format="star",       # *, **, *** notation
        loc="inside",            # place above bars
        comparisons_correction=None  # no correction
    )

    annotator.apply_and_annotate()
    plt.title('Intensity Ratio between nuclei and cytoplasm by Sample')
    plt.xlabel('Sample')
    plt.ylabel('Ratio')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plot_path = output_dir / "intensity_ratio_barplot_std.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")