# Train a Cellpose model on your corrected masks.

from pathlib import Path
import numpy as np
import tifffile
from cellpose import models, io

def train_cellpose_model(
    image_dir: Path,
    mask_dir: Path,
    model_name: str = "custom_cellpose_model",
    channels: list = [1, 2],  # [cyto_channel, nuc_channel]
    n_epochs: int = 100,
    learning_rate: float = 0.1,
    batch_size: int = 8,
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
    image_paths = sorted(image_dir.glob("*ch1.tiff"))  # Adjust channel as needed
    mask_paths = sorted(mask_dir.glob("*_masks.tiff"))

    if not image_paths or not mask_paths:
        raise FileNotFoundError(
            f"Images: {len(image_paths)} found in {image_dir}\n"
            f"Masks: {len(mask_paths)} found in {mask_dir}"
        )

    # Verify matching pairs
    if len(image_paths) != len(mask_paths):
        print(f"Warning: {len(image_paths)} images but {len(mask_paths)} masks")

    # Load images and masks into memory
    images = [tifffile.imread(p) for p in image_paths]
    masks = [tifffile.imread(p) for p in mask_paths]

    print(f" Loaded {len(images)} images and {len(masks)} masks")
    print(f"  Sample image shape: {images[0].shape}")
    print(f"  Sample mask shape: {masks[0].shape}")

    # Initialize model (use existing Cellpose model as base)
    print(f"\nTraining Cellpose model '{model_name}'...")
    print(f"  Epochs: {n_epochs}, LR: {learning_rate}, Batch size: {batch_size}")

    model = models.CellposeModel(
        gpu=True,
        pretrained_model='cpsam',  # Start from pretrained cyto model
        model_type=None,  # Will use the pretrained model
    )

    # Train the model
    new_model_path = model.train(
        images,
        masks,
        channels=channels,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_name=model_name,
    )

    print(f" Model trained and saved to: {new_model_path}")
    return model

