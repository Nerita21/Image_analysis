"""
Measure intensity of green channel in nuclear, cytoplasmic, and granular regions.
"""

from pathlib import Path
import numpy as np
import tifffile
from skimage import measure, filters
import pandas as pd
from cellpose import models
from typing import Optional


def extract_intensity_measurements(
    cyto_image: np.ndarray,
    nuc_image: np.ndarray,
    cyto_mask: np.ndarray,
    nuc_mask: np.ndarray,
    granule_threshold: float = 0.75,
    signal_channel: Optional[np.ndarray] = None,
    green_channel: Optional[np.ndarray] = None,
    bg_subtract: bool = True,
    normalize: str = "none",  # options: 'none', 'max', 'percentile'
    percentile: float = 99.0,
) -> dict:
    """
    Extract intensity measurements from segmented regions.

    Args:
        cyto_image (np.ndarray): Cytoplasm channel image (for segmentation reference)
        nuc_image (np.ndarray): Nuclear channel image (for segmentation reference)
        green_channel (Optional[np.ndarray]): Optional explicit green channel to measure from; if None, cyto_image is used
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

    # Choose intensity channel (priority: explicit signal_channel, green_channel, then cyto_image)
    if signal_channel is not None:
        intensity_channel = signal_channel
    elif green_channel is not None:
        intensity_channel = green_channel
    else:
        intensity_channel = cyto_image

    # Ensure float for processing
    intensity_channel = intensity_channel.astype(float)

    # Background subtraction (default: median of pixels outside masks)
    intensity_bg = intensity_channel.copy()
    if bg_subtract:
        bg_mask = ~( (cyto_mask > 0) | (nuc_mask > 0) )
        if np.any(bg_mask):
            bg_val = float(np.median(intensity_channel[bg_mask]))
        else:
            bg_val = float(np.median(intensity_channel))
        intensity_bg = intensity_channel - bg_val
        intensity_bg[intensity_bg < 0] = 0.0

    # Normalization for detection (keeps reporting on bg-subtracted values)
    intensity_normalized = intensity_bg.copy()
    if normalize == "max":
        max_val = float(np.max(intensity_bg)) if intensity_bg.size else 0.0
        if max_val > 0:
            intensity_normalized = intensity_bg / max_val
    elif normalize == "percentile":
        pval = float(np.percentile(intensity_bg, percentile)) if intensity_bg.size else 0.0
        if pval > 0:
            intensity_normalized = intensity_bg / pval

    # ===== NUCLEAR INTENSITY =====
    if np.any(nuc_mask_binary):
        nuclear_region = intensity_bg[nuc_mask_binary]
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
        cyto_region = intensity_bg[cyto_only_mask]
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
        # Find high-intensity spots in cytoplasm using the normalized (bg-subtracted) image
        granule_threshold_val = granule_threshold * np.max(intensity_normalized[cyto_only_mask])
        granule_mask = (intensity_normalized > granule_threshold_val) & cyto_only_mask

        if np.any(granule_mask):
            # Label connected components (individual granules)
            granule_labels = measure.label(granule_mask)
            num_granules = len(np.unique(granule_labels)) - 1  # Exclude background (0)

            measurements["granule_count"] = int(num_granules)
            measurements["granule_intensity_mean"] = float(np.mean(intensity_channel[granule_mask]))
            measurements["granule_intensity_max"] = float(np.max(intensity_channel[granule_mask]))

            # Detailed granule info
            granule_list = []
            for gid in np.unique(granule_labels)[1:]:  # Skip 0 (background)
                granule_pixels = intensity_channel[granule_labels == gid]
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

def _find_file_by_pattern(directory: Path, base_name: str, suffix: str, extensions: list = [".tiff", ".tif"]) -> Optional[Path]:
    """
    Find a file in directory matching base_name + suffix, accepting multiple extensions.
    
    Args:
        directory: Path to search in
        base_name: Base name (without extension or suffix)
        suffix: Suffix to match (e.g., "_ch1", "_mask")
        extensions: List of extensions to try (e.g., [".tiff", ".tif"])
    
    Returns:
        Path to file if found, None otherwise
    """
    for ext in extensions:
        candidate = directory / f"{base_name}{suffix}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_intensity_batch(
       image_dir: Path,
       output_csv: Path,
       mask_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Process all intensity images organized in subfolders and find matching masks.
    
    Uses the subfolder name as the sample ID in the CSV.
    Matches images by base name (ignoring _ch1, _ch2, _masks suffixes and extensions).
    Supports both .tiff and .tif extensions.

    Args:
        image_dir (Path): Parent directory containing sample subfolders (each with *_ch1.tiff, etc.)
        output_csv (Path): Where to save results CSV
        mask_dir (Optional[Path]): Parent directory containing mask subfolders. 
                                    If None, assumes same structure as image_dir

    Returns:
        pd.DataFrame: Results table with all measurements
    """
    results = []
    
    if mask_dir is None:
        mask_dir = image_dir

    # Find all sample subfolders
    sample_folders = sorted([f for f in image_dir.iterdir() if f.is_dir()])
    
    print(f"Processing {len(sample_folders)} samples from {image_dir}...")

    for sample_folder in sample_folders:
        sample_name = sample_folder.name  # Use subfolder name as sample ID
        
        # Find mask subfolder (same name in mask_dir)
        mask_subfolder = mask_dir / sample_name
        if not mask_subfolder.exists():
            mask_subfolder = mask_dir  # Fallback: look for masks in root mask_dir
        
        # Find all ch1 intensity images in this sample folder (both .tiff and .tif)
        ch1_files = sorted(sample_folder.glob("*_ch1.tiff")) + sorted(sample_folder.glob("*_ch1.tif"))
        ch1_files = list(dict.fromkeys(ch1_files))  # Remove duplicates while preserving order


        for ch1_file in ch1_files:
            # Extract base name: e.g., "image1_ch1.tiff" → "image1"
            base_name = ch1_file.stem.replace("_ch1", "")
            try:
                # Load intensity channels
                ch1 = tifffile.imread(ch1_file)

                # Try to find ch2
                ch2_file = _find_file_by_pattern(sample_folder, base_name, "_ch2")
                ch2 = None
                if ch2_file:
                    ch2 = tifffile.imread(ch2_file)

                # Find and load cytoplasmic mask: try _ch1_masks, then _masks
                cyto_mask_patterns = ["_ch1_masks", "_masks"]
                cyto_mask_file = None
                tried_cyto_patterns = []
                for pat in cyto_mask_patterns:
                    cyto_mask_file = _find_file_by_pattern(mask_subfolder, base_name, pat)
                    tried_cyto_patterns.append(f"{base_name}{pat}.*")
                    if cyto_mask_file:
                        break
                if not cyto_mask_file:
                    print(f"    ⚠ {sample_name}/{base_name}: No cytoplasmic mask file found (tried: {', '.join(tried_cyto_patterns)})")
                    continue
                cyto_mask = tifffile.imread(cyto_mask_file)

                # For nuclear mask, try _ch2_masks (if ch2 exists), then _nuc_masks
                nuc_mask_patterns = []
                if ch2 is not None:
                    nuc_mask_patterns.append("_ch2_masks")
                nuc_mask_patterns.append("_nuc_masks")
                nuc_mask_file = None
                tried_nuc_patterns = []
                for pat in nuc_mask_patterns:
                    nuc_mask_file = _find_file_by_pattern(mask_subfolder, base_name, pat)
                    tried_nuc_patterns.append(f"{base_name}{pat}.*")
                    if nuc_mask_file:
                        break
                if nuc_mask_file:
                    nuc_mask = tifffile.imread(nuc_mask_file)
                else:
                    nuc_mask = np.zeros_like(cyto_mask)

                # Measure intensity (ch1 = green cyto signal)
                meas = extract_intensity_measurements(
                    cyto_image=ch1,
                    nuc_image=ch2 if ch2 is not None else ch1,
                    cyto_mask=cyto_mask,
                    nuc_mask=nuc_mask,
                    signal_channel=ch1,  # ch1 is the green channel with signal
                )

                # Flatten for CSV (sample_name is the subfolder, base_name is the image within)
                row = {"sample_name": sample_name, "image_name": base_name, **meas}
                results.append(row)
                print(f"    ✓ {sample_name}/{base_name}")

            except Exception as e:
                print(f"    ✗ {sample_name}/{base_name}: {e}")
                continue

    # Create DataFrame and save
    if not results:
        print("⚠ No images processed! Check your directories and file names.")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to {output_csv}")

    return df

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


