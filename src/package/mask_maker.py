from pathlib import Path
import tifffile
import numpy as np
from cellpose import models

def create_masks(tiff_dir: Path, model: models.CellposeModel, channel: str | None = None) -> None:
    """
    Process all TIFF images in the specified directory using Cellpose,
    generate masks, and save them next to the original TIFF files.

    Args:
        tiff_dir (Path): Directory containing TIFF files to process.
        model (models.CellposeModel): Pre-initialized Cellpose model.
        channel (str, optional): String pattern to match channel name (e.g., 'DAPI' or 'ch1').
                                 If None, all TIFF files in the directory are processed.
    """
    # Select files depending on whether a channel name is provided
    if channel:
        tiff_files = sorted(tiff_dir.rglob(f"*{channel}*.tiff"))
    else:
        tiff_files = sorted(tiff_dir.rglob("*.tiff"))

    if not tiff_files:
        print(f"No TIFF files found in {tiff_dir} for channel={channel!r}")
        return

    for tiff_file in tiff_files:
        mask_file = tiff_file.with_name(f"{tiff_file.stem}_mask.tiff")

        if mask_file.exists():
            print(f"Mask already exists for {tiff_file.name} — skipping.")
            continue

        print(f"Processing {tiff_file.name}...")
        image = tifffile.imread(tiff_file)

        # Call the Cellpose model and unpack the results
        # model.eval returns (masks, flows, styles, diams)
        masks, _, _, _ = model.eval(image, diameter=None, channels=[channel, 0])

        # Define output path for the mask
        mask_file = tiff_dir / f"{tiff_file.stem}_mask.tiff"

        # Save mask (ensure masks is a numpy array before casting type)
        tifffile.imwrite(mask_file, masks.astype(np.uint16))
        print(f"Saved mask to {mask_file.relative_to(tiff_dir)}")