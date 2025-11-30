# src/package/tiff_viewer.py
import tifffile
import numpy as np
import napari
from pathlib import Path
import re
from collections import defaultdict
def view_tiff_directory(tiff_dir: Path, group_channels: bool = True) -> None:
    """
    Open TIFF images in the specified directory with Napari.
    Handles both:
      - Multi-channel TIFFs (single file containing multiple channels)
      - Separate TIFFs for each channel (e.g. *_ch1.tiff, *_ch2.tiff)

    Args:
        tiff_dir (Path): Directory containing .tiff files.
        group_channels (bool): If True, groups separate per-channel TIFFs
                               by their base name and adds them as a single
                               multi-channel layer in Napari.
    """
    tiff_files = sorted(tiff_dir.glob("*.tiff"))
    if not tiff_files:
        print(f"No TIFF files found in {tiff_dir}")
        return

    viewer = napari.Viewer()

    if group_channels:
        # Group files by base name without _chX suffix
        channel_pattern = re.compile(r"(.+)_ch(\d+)$")
        groups = defaultdict(list)

        for f in tiff_files:
            m = channel_pattern.match(f.stem)
            if m:
                base, ch_num = m.groups()
                groups[base].append((int(ch_num), f))
            else:
                # No _chX pattern — treat as standalone file
                groups[f.stem].append((1, f))

        for base_name, files in groups.items():
            # Sort by channel number
            files.sort(key=lambda x: x[0])
            imgs = []
            for ch_num, f in files:
                img = tifffile.imread(f)
                imgs.append(img)

            if len(imgs) == 1:
                # Single channel file — could still be multi-channel internally
                img = imgs[0]
                if img.ndim >= 3 and img.shape[0] <= 4:  # Heuristic: small first axis = channels
                    viewer.add_image(img, name=base_name, channel_axis=0)
                else:
                    viewer.add_image(img, name=base_name)
            else:
                # Stack separate per-channel TIFFs
                stacked = np.stack(imgs, axis=0)  # shape: (C, ...)
                viewer.add_image(stacked, name=base_name, channel_axis=0)

    else:
        # Load each TIFF file independently, auto-detect channel axis
        for f in tiff_files:
            img = tifffile.imread(f)
            # Heuristic: if first axis is small (e.g. 2–4), likely channel axis
            if img.ndim >= 3 and 1 < img.shape[0] <= 4:
                viewer.add_image(img, name=f.stem, channel_axis=0)
            else:
                viewer.add_image(img, name=f.stem)

    napari.run()

def view_masks_napari(mask_dir: Path, tiff_dir: Path, mask_suffix: str = "_mask") -> None:
    """
    View raw TIFFs + mask TIFFs in Napari.
    Mask files end with `_mask.tiff` and raw files do NOT have the suffix.
    """

    mask_files = sorted(mask_dir.glob(f"*{mask_suffix}.tiff"))
    if not mask_files:
        print(f"No mask TIFF files found in {mask_dir} with suffix {mask_suffix}")
        return

    viewer = napari.Viewer()

    for mask_file in mask_files:
        # Example: "sample1_mask" → "sample1"
        base = mask_file.stem.replace(mask_suffix, "")

        # Raw TIFF with same base name
        raw_file = tiff_dir / f"{base}.tiff"
        if not raw_file.exists():
            print(f"No raw TIFF found for mask {mask_file.name}")
            continue

        # Load both images
        mask = tifffile.imread(mask_file)
        raw = tifffile.imread(raw_file)

        # Add to Napari
        viewer.add_image(raw, name=f"{base}_raw")
        viewer.add_labels(mask, name=f"{base}_mask")

    napari.run()
