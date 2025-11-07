from pathlib import Path
import tifffile
import nd2
def convert_nd2_to_tiff(nd2_dir: Path, tiff_dir: Path) -> None:
    """
    Recursively search for ND2 files under nd2_dir and convert them to TIFF.
    For each ND2 file, check how many channels it has. If all expected TIFF files
    already exist in the output directory, skip conversion. If any are missing,
    generate all TIFF files for that ND2.

    Args:
        nd2_dir (Path): Root directory containing .nd2 image files (may include subfolders).
        tiff_dir (Path): Directory where TIFF files will be saved.
    """
    tiff_dir.mkdir(parents=True, exist_ok=True)

    # Recursively search for .nd2 files
    nd2_files = list(nd2_dir.rglob("*.nd2"))

    if not nd2_files:
        print(f"No ND2 files found in {nd2_dir}")
        return

    for nd2_file in nd2_files:
        print(f"\nChecking {nd2_file.relative_to(nd2_dir)}...")

        # Open file to check channel count (quick metadata check)
        with nd2.ND2File(nd2_file) as img:
            n_channels = img.sizes.get("C", 1)
            img_data = None

        # Build output path to mirror folder structure relative to nd2_dir
        relative_subdir = nd2_file.parent.relative_to(nd2_dir)
        output_subdir = tiff_dir / relative_subdir
        output_subdir.mkdir(parents=True, exist_ok=True)

        expected_tiffs = [
            output_subdir / f"{nd2_file.stem}_ch{ch+1}.tiff"
            for ch in range(n_channels)
        ]

        # Check if all expected TIFFs already exist
        if all(tiff_path.exists() for tiff_path in expected_tiffs):
            print(f"✔ All {n_channels} TIFFs exist — skipping {nd2_file.name}")
            continue

        # Convert and save TIFFs
        print(f"→ Some TIFFs missing — converting {nd2_file.name}...")
        with nd2.ND2File(nd2_file) as img:
            img_data = img.asarray()  # shape: (frames, channels, height, width)

        for ch in range(n_channels):
            tiff_path = expected_tiffs[ch]
            tifffile.imwrite(tiff_path, img_data[ch, :, :])
            print(f"Saved {tiff_path.relative_to(tiff_dir)}")

    print("\n Recursive ND2 to TIFF conversion completed.")