from pathlib import Path
import tifffile
import numpy as np
from cellpose import models, utils, dynamics

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
        # model.eval returns (masks, flows, styles, diams). We only need the masks.
        result = model.eval(image, diameter=None, flow_threshold=None)
        masks = result[0] if isinstance(result, (list, tuple)) else result

        
        # Save mask (ensure masks is a numpy array before casting type)
        tifffile.imwrite(mask_file, masks.astype(np.uint16))
        print(f"Saved mask to {mask_file.relative_to(tiff_dir)}")

def create_flows(mask_folder_path):
    """
    Loads mask files from a folder, computes Cellpose flows, 
    and saves them as .npy files next to each mask.

    Args:
        mask_folder_path (str or Path): folder containing *_mask.tif images
    """

    mask_folder_path = Path(mask_folder_path)

    # Find all mask files
    mask_files = list(mask_folder_path.glob("*_mask*.tif"))

    if len(mask_files) == 0:
        raise FileNotFoundError(f"No mask images found in: {mask_folder_path}")

    for mask_file in mask_files:
        print(f"Processing: {mask_file.name}")

        # Load mask
        mask = utils.imread(mask_file)
        mask_int = mask.astype(np.int32)

        # Compute flows
        flows = dynamics.compute_flows(mask_int)

        # Save output as .npy
        out_path = mask_file.with_suffix(".npy")  # e.g., image_mask.tif → image_mask.npy
        np.save(out_path, {"masks": mask_int, "flows": flows})

        print(f"Saved: {out_path}")
