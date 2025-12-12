
from utils import load_config
from package import process_image_batch
from cellpose import models
from pathlib import Path

def run_intensity_pipeline(image_dir: str, output_csv: str, model_name: str):
    """
    Automates measurement of nuclear/cytoplasmic intensity from images.
    
    Args:
        image_dir (str): directory containing input TIFF images
        output_csv (str): path to save measurements CSV
        model_name (str): name of the pretrained Cellpose model to use
    """

    image_dir = Path(image_dir)
    output_csv = Path(output_csv)

    print(f"\nLoading model '{model_name}'...")
    model = models.CellposeModel(pretrained_model=model_name, gpu=True)

    print(f"Processing images in: {image_dir}")
    df = process_image_batch(
        image_dir=image_dir,
        model=model,
        output_csv=output_csv,
    )

    print(f"\nResults saved to: {output_csv}")

    print("\n=== Summary Statistics ===")
    print(df[[
        "nuclear_intensity_mean",
        "cytoplasmic_intensity_mean",
        "nuclear_to_cytoplasmic_ratio"
    ]].describe())

    return df

