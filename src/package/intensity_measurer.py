"""
Measure intensity of green channel in nuclear, cytoplasmic, and granular regions.
"""

from pathlib import Path
import numpy as np
import tifffile
from skimage import measure, filters
import pandas as pd
from cellpose import models


def extract_intensity_measurements(
    cyto_image: np.ndarray,
    nuc_image: np.ndarray,
    green_channel: np.ndarray,
    cyto_mask: np.ndarray,
    nuc_mask: np.ndarray,
    granule_threshold: float = 0.75,
) -> dict:
    """
    Extract intensity measurements from segmented regions.

    Args:
        cyto_image (np.ndarray): Cytoplasm channel image (for segmentation reference)
        nuc_image (np.ndarray): Nuclear channel image (for segmentation reference)
        green_channel (np.ndarray): The green channel to measure intensity from
        cyto_mask (np.ndarray): Binary or labeled mask for cytoplasm regions
        nuc_mask (np.ndarray): Binary or labeled mask for nuclear regions
        granule_threshold (float): Threshold for granule detection (0-1 scale relative to max)

    Returns:
        dict: Dictionary containing measurements:
            - nuclear_intensity_mean
            - nuclear_intensity_std
            - cytoplasmic_intensity_mean
            - cytoplasmic_intensity_std
            - granule_count (number of granules detected)
            - granule_intensity_mean
            - nuclear_to_cytoplasmic_ratio
            - granule_list (list of granule info dicts)
    """
    measurements = {}

    # Ensure masks are binary
    cyto_mask_binary = cyto_mask > 0
    nuc_mask_binary = nuc_mask > 0

    # Normalize green channel for threshold
    green_normalized = green_channel / np.max(green_channel)

    # ===== NUCLEAR INTENSITY =====
    if np.any(nuc_mask_binary):
        nuclear_region = green_channel[nuc_mask_binary]
        measurements["nuclear_intensity_mean"] = float(np.mean(nuclear_region))
        measurements["nuclear_intensity_std"] = float(np.std(nuclear_region))
        measurements["nuclear_intensity_max"] = float(np.max(nuclear_region))
        measurements["nuclear_intensity_median"] = float(np.median(nuclear_region))
    else:
        measurements["nuclear_intensity_mean"] = 0.0
        measurements["nuclear_intensity_std"] = 0.0
        measurements["nuclear_intensity_max"] = 0.0
        measurements["nuclear_intensity_median"] = 0.0

    # ===== CYTOPLASMIC INTENSITY =====
    # Cytoplasm = cytoplasm mask - nuclear mask (exclude nucleus from cyto)
    cyto_only_mask = cyto_mask_binary & ~nuc_mask_binary

    if np.any(cyto_only_mask):
        cyto_region = green_channel[cyto_only_mask]
        measurements["cytoplasmic_intensity_mean"] = float(np.mean(cyto_region))
        measurements["cytoplasmic_intensity_std"] = float(np.std(cyto_region))
        measurements["cytoplasmic_intensity_max"] = float(np.max(cyto_region))
        measurements["cytoplasmic_intensity_median"] = float(np.median(cyto_region))
    else:
        measurements["cytoplasmic_intensity_mean"] = 0.0
        measurements["cytoplasmic_intensity_std"] = 0.0
        measurements["cytoplasmic_intensity_max"] = 0.0
        measurements["cytoplasmic_intensity_median"] = 0.0

    # ===== GRANULE DETECTION (puncta in cytoplasm) =====
    if np.any(cyto_only_mask):
        # Find high-intensity spots in cytoplasm
        granule_threshold_val = granule_threshold * np.max(green_normalized[cyto_only_mask])
        granule_mask = (green_normalized > granule_threshold_val) & cyto_only_mask

        if np.any(granule_mask):
            # Label connected components (individual granules)
            granule_labels = measure.label(granule_mask)
            num_granules = len(np.unique(granule_labels)) - 1  # Exclude background (0)

            measurements["granule_count"] = int(num_granules)
            measurements["granule_intensity_mean"] = float(np.mean(green_channel[granule_mask]))
            measurements["granule_intensity_max"] = float(np.max(green_channel[granule_mask]))

            # Detailed granule info
            granule_list = []
            for gid in np.unique(granule_labels)[1:]:  # Skip 0 (background)
                granule_pixels = green_channel[granule_labels == gid]
                granule_list.append({
                    "granule_id": int(gid),
                    "intensity_mean": float(np.mean(granule_pixels)),
                    "intensity_max": float(np.max(granule_pixels)),
                    "pixel_count": int(len(granule_pixels)),
                })
            measurements["granule_list"] = granule_list
        else:
            measurements["granule_count"] = 0
            measurements["granule_intensity_mean"] = 0.0
            measurements["granule_intensity_max"] = 0.0
            measurements["granule_list"] = []
    else:
        measurements["granule_count"] = 0
        measurements["granule_intensity_mean"] = 0.0
        measurements["granule_intensity_max"] = 0.0
        measurements["granule_list"] = []

    # ===== RATIOS =====
    if measurements["cytoplasmic_intensity_mean"] > 0:
        measurements["nuclear_to_cytoplasmic_ratio"] = (
            measurements["nuclear_intensity_mean"] / measurements["cytoplasmic_intensity_mean"]
        )
    else:
        measurements["nuclear_to_cytoplasmic_ratio"] = 0.0

    return measurements


def process_image_batch(
    image_dir: Path,
    model: models.CellposeModel,
    output_csv: Path,
    model_ch1: int = 1,
    model_ch2: int = 0,
) -> pd.DataFrame:
    """
    Process all images in a directory: segment + measure intensity.

    Args:
        image_dir (Path): Directory with images (*_ch1.tiff, *_ch2.tiff, etc.)
        model (CellposeModel): Trained or pretrained Cellpose model
        output_csv (Path): Where to save results CSV
        model_ch1 (int): Which channel index for primary segmentation (0 or 1)
        model_ch2 (int): Which channel index for secondary (0=unused)

    Returns:
        pd.DataFrame: Results table with all measurements
    """
    results = []

    # Find all ch1 images
    ch1_files = sorted(image_dir.glob("*ch1.tiff"))

    print(f"Processing {len(ch1_files)} images...")

    for ch1_file in ch1_files:
        image_name = ch1_file.stem.replace("_ch1", "")

        try:
            # Load channels
            ch1 = tifffile.imread(ch1_file)
            ch2_file = ch1_file.with_name(f"{image_name}_ch2.tiff")

            if ch2_file.exists():
                ch2 = tifffile.imread(ch2_file)
            else:
                ch2 = None

            # Segment using Cellpose
            print(f"  Segmenting {image_name}...")
            if ch2 is not None:
                # Two-channel input to model
                masks = model.eval(
                    [ch1, ch2],
                    channels=[model_ch1, model_ch2],
                    diameter=None,
                )
            else:
                # Single channel
                masks = model.eval(ch1, diameter=None)

            # For this example, assume:
            # - masks[0] = cytoplasm segmentation
            # - masks[1] = nuclear segmentation (optional)
            cyto_mask = masks[0] if isinstance(masks, tuple) else masks

            # Create a nuclear mask (simple heuristic: nuclei are in ch2)
            if ch2 is not None:
                nuc_masks = model.eval(ch2, diameter=None)
                nuc_mask = nuc_masks[0] if isinstance(nuc_masks, tuple) else nuc_masks
            else:
                nuc_mask = np.zeros_like(cyto_mask)

            # Measure intensity (green channel is typically ch1, adjust as needed)
            meas = extract_intensity_measurements(
                cyto_image=ch1,
                nuc_image=ch2 if ch2 is not None else ch1,
                green_channel=ch1,  # Adjust if green is ch2
                cyto_mask=cyto_mask,
                nuc_mask=nuc_mask,
            )

            # Flatten for CSV
            row = {"image_name": image_name, **meas}
            results.append(row)
            print(f"    ✓ {image_name}")

        except Exception as e:
            print(f"    ✗ {image_name}: {e}")
            continue

    # Create DataFrame and save
    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to {output_csv}")

    return df


if __name__ == "__main__":
    from utils import load_config

    config, base_dir = load_config()

    # Example: measure on test images
    image_dir = base_dir / config["data"]["raw_tiff_images_dir"]
    output_csv = base_dir / "data/csv/intensity_measurements.csv"

    # Use pretrained model for testing
    model = models.CellposeModel(pretrained_model='cyto', gpu=True)

    df = process_image_batch(
        image_dir=image_dir,
        model=model,
        output_csv=output_csv,
    )

    print("\n=== Summary Statistics ===")
    print(df[["nuclear_intensity_mean", "cytoplasmic_intensity_mean",
              "nuclear_to_cytoplasmic_ratio"]].describe())
