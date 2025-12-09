# Train a Cellpose model on your corrected masks.

from pathlib import Path
import numpy as np
import tifffile
from cellpose import models, io, train

def train_cellpose_model(
    image_dir: Path,
    mask_dir: Path,
    test_dir: Path = None,
    model_name: str = "custom_cellpose_model",
    channels: list = [1, 2],  # [cyto_channel, nuc_channel]
    n_epochs: int = 100,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
) -> models.CellposeModel:
    """
    Train a Cellpose model using corrected segmentation masks.

    Args:
        image_dir (Path): Directory containing training images (.tiff files)
        mask_dir (Path): Directory containing corrected masks (*_masks.tiff files)
        model_name (str): Name for your trained model (saved to models/model_name)
        channels (list): [cyto_channel, nuclear_channel] (typically [1, 2])
                        Set to [0, 0] for single-channel training
        n_epochs (int): Number of training epochs (100-500 recommended)
        learning_rate (float): Learning rate for training (0.1 is default)
        batch_size (int): Batch size (default 8, reduce if OOM)

    Returns:
        CellposeModel: The trained model object
    """
    # Load image and mask files
    image_paths = get_image_paths(image_dir, channels)
    mask_paths = sorted(
        list(mask_dir.glob("*_masks.tiff")) +
        list(mask_dir.glob("*_masks.tif"))
    )
    
    if len(channels) == 1:
        ch_pattern = f"*ch{channels[0]}*"
    else:
        ch_pattern = "*ch*"

    if len(channels) == 1 or set(channels) == {2}:
        train_channels = [0,0]
    else:
        train_channels = channels



    if not image_paths or not mask_paths:
        raise FileNotFoundError(
            f"Images: {len(image_paths)} found in {image_dir}\n"
            f"Masks: {len(mask_paths)} found in {mask_dir}"
        )

    # Verify matching pairs
    if len(image_paths) != len(mask_paths):
        print(f"Warning: {len(image_paths)} images but {len(mask_paths)} masks")

    # Load images and masks into memory
    # images = [tifffile.imread(p) for p in image_paths]
    # masks = [tifffile.imread(p) for p in mask_paths]

    # Initialize model (use existing Cellpose model as base)
    print(f"\nTraining Cellpose model '{model_name}'...")
    print(f"  Epochs: {n_epochs}, LR: {learning_rate}, Batch size: {batch_size}")

    io.logger_setup()

    output = io.load_train_test_data(
        train_dir=str(image_dir),
        test_dir=str(test_dir) if test_dir else None,
        image_filter=ch_pattern,
        mask_filter="_masks",
        look_one_level_down=False
    )

    images, labels, _, test_images, test_labels, _ = output

    model = models.CellposeModel(
        gpu=True,
        pretrained_model='cpsam', 
        model_type=None,  # Will use the pretrained model
    )

    # Train the model (new version)
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        test_data=test_images,
        test_labels=test_labels,
        learning_rate=learning_rate,
        weight_decay=0.1,
        channels=train_channels,
        n_epochs=n_epochs,
        model_name=model_name
    )

    print(f" Model trained and saved to: {model_path}")
    return model

def get_image_paths(image_dir: Path, channels: list):
    """Return image paths depending on channel selection."""

    # normalize channels (remove zeros)
    channels = [c for c in channels if c > 0]

    # CASE 1: two channels → load both ch1 and ch2
    if len(channels) == 2 and set(channels) == {1, 2}:
        patterns = ["*ch1.tif", "*ch1.tiff", "*ch2.tif", "*ch2.tiff"]

    # CASE 2: single channel → load only that channel
    elif len(channels) == 1:
        ch = channels[0]
        patterns = [f"*ch{ch}.tif", f"*ch{ch}.tiff"]

    # CASE 3: channels=[0,0] → load all tifs
    else:
        patterns = ["*.tif", "*.tiff"]

    # Glob all patterns
    files = []
    for pat in patterns:
        files += list(image_dir.glob(pat))

    return sorted(files)
