# Train a Cellpose model on your corrected masks.

from pathlib import Path
import numpy as np
import tifffile
from cellpose import models, io, train
from skimage.measure import label

def train_cellpose_model(
    image_dir: Path,
    mask_dir: Path,
    test_dir: Path = None,
    model_name: str = "custom_cellpose_model",
    channel_id: str = None,
    gpu: bool = True,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 1,
) -> models.CellposeModel:
    """
    Train a Cellpose model using corrected segmentation masks.

    Args:
        image_dir (Path): Directory containing training images (.tiff files)
        mask_dir (Path): Directory containing corrected masks (*_masks.tiff files)
        model_name (str): Name for your trained model (saved to models/model_name)
        test_dir (Path, optional): Directory with test images and masks for validation
        channel_id (str, optional): Channel identifier in filenames (e.g., "ch2")
        n_epochs (int): Number of training epochs (100-500 recommended)
        learning_rate (float): Learning rate for training (0.1 is default)
        batch_size (int): Batch size (default 8, reduce if OOM)

    Returns:
        CellposeModel: The trained model object
    """
    # Load image and mask files
    image_paths = get_image_paths(image_dir, channel_id)
    mask_paths = sorted(
        list(mask_dir.glob("*_masks.tiff")) +
        list(mask_dir.glob("*_masks.tif")) +
        list(mask_dir.glob("*_mask.tiff")) +
        list(mask_dir.glob("*_mask.tif"))
    )
    
    

    if not image_paths:
        raise FileNotFoundError(
            f"Images: {len(image_paths)} found in {image_dir}\n"
        )
    
    if not mask_paths:
        raise FileNotFoundError(
            f"Masks: {len(mask_paths)} found in {mask_dir}"
        )

    # Verify matching pairs
    if len(image_paths) != len(mask_paths):
        print(f"Warning: {len(image_paths)} images but {len(mask_paths)} masks")

    # Load images and masks into memory
    images, labels, test_images, test_labels = load_paired_images_and_masks(
        image_paths,
        mask_paths,
        test_dir=test_dir,
        channel_id=channel_id,
    )

    print(f"Loaded {len(images)} training images and {len(labels)} training labels")
    if not labels:
        print("ERROR: No labels (masks) found by Cellpose loader!")
        print("Check mask naming (should end with '_mask.tiff' or '_masks.tiff') and directory structure.")
        print(f"Mask directory: {mask_dir}")
        print(f"Sample mask files found: {mask_paths[:5]}")
        raise ValueError("No training labels found. Ensure masks are properly named and placed.")

    model = models.CellposeModel(
        gpu=gpu,
        pretrained_model='cpsam', 
        model_type=None,  # Will use the pretrained model
    )

    # Preprocess masks to remove border objects and relabel
    labels = [process_mask_for_cellpose(label) for label in labels]

    pairs = [
        (img, lbl)
        for img, lbl in zip(images, labels)
        if lbl is not None and lbl.max() > 0
    ]

    if not pairs:
        raise ValueError("All training masks are empty after preprocessing")

    images, labels = zip(*pairs)

    batch_size = min(batch_size, len(images))
   
    # Train the model (new version)
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        test_data=test_images,
        test_labels=test_labels,
        learning_rate=learning_rate,
        weight_decay=0.1,
        batch_size=batch_size,
        n_epochs=n_epochs,
        model_name=model_name
    )

    print(f" Model trained and saved to: {model_path}")
    return model

def get_image_paths(image_dir: Path, channel_id: str = None):
    """
    Return image paths filtered by channel identifier in filename.

    Examples:
        channel_id="ch2"  -> *_ch2.tif*
        channel_id=None   -> all .tif/.tiff files
    """

    if channel_id:
        patterns = [
            f"*{channel_id}*.tif",
            f"*{channel_id}*.tiff",
        ]
    else:
        patterns = ["*.tif", "*.tiff"]

    files = []
    for pat in patterns:
        files.extend(image_dir.glob(pat))

    return sorted(files)


def load_paired_images_and_masks(
    image_paths,
    mask_paths,
    test_dir=None,
    channel_id=None,
):
    """
    Load images and their corresponding masks into memory by matching filenames.

    Parameters
    ----------
    image_paths : list[Path]
        List of image file paths.
    mask_paths : list[Path]
        List of mask file paths.
    test_dir : str or Path, optional
        Directory containing test images and masks.
    channels : list, optional
        Channel identifiers used by get_image_paths() for test data.

    Returns
    -------
    images : list[np.ndarray]
        Paired training images.
    labels : list[np.ndarray]
        Paired training masks.
    test_images : list[np.ndarray]
        Paired test images (empty if test_dir not provided).
    test_labels : list[np.ndarray]
        Paired test masks (empty if test_dir not provided).
    """

    images = []
    labels = []
    paired_count = 0

    # ---- Load training images and masks ----
    for img_path in image_paths:
        base = img_path.stem
        mask_path = None

        for m_path in mask_paths:
            m_base = m_path.stem.replace('_masks', '').replace('_mask', '')
            if m_base == base:
                mask_path = m_path
                break

        if mask_path:
            images.append(tifffile.imread(img_path))
            labels.append(tifffile.imread(mask_path))
            paired_count += 1
        else:
            print(f"Warning: No matching mask for image {img_path.name}")

    print(
        f"Paired {paired_count} image-mask pairs out of "
        f"{len(image_paths)} images and {len(mask_paths)} masks"
    )

    if not images or not labels:
        raise ValueError("No paired image-mask data found. Check naming conventions.")

    # ---- Load test images and masks (optional) ----
    if test_dir is not None:
        test_images = []
        test_labels = []

        if test_dir and Path(test_dir).exists():
            test_dir = Path(test_dir)

            test_image_paths = get_image_paths(test_dir, channel_id)
            test_mask_paths = sorted(
                list(test_dir.glob("*_masks.tiff")) +
                list(test_dir.glob("*_masks.tif")) +
                list(test_dir.glob("*_mask.tiff")) +
                list(test_dir.glob("*_mask.tif"))
            )

            test_mask_dict = {
                m.stem.replace('_masks', '').replace('_mask', ''): m
                for m in test_mask_paths
            }

            for img_path in test_image_paths:
                base = img_path.stem
                if base in test_mask_dict:
                    test_images.append(tifffile.imread(img_path))
                    test_labels.append(tifffile.imread(test_mask_dict[base]))
    # If no test data, set to None to avoid index errors
    if not test_images:
        test_images = None
        test_labels = None

    return images, labels, test_images, test_labels

def process_mask_for_cellpose(mask: np.ndarray) -> np.ndarray:
    """
    Prepare instance mask for Cellpose training.
    - Removes border-touching objects
    - Relabels instances consecutively starting from 1
    """
    mask = mask.astype(np.int32)

    # Identify border-touching labels
    border_labels = np.unique(np.concatenate([
        mask[0, :],
        mask[-1, :],
        mask[:, 0],
        mask[:, -1]
    ]))
    border_labels = border_labels[border_labels > 0]

    # Remove border objects
    mask_clean = mask.copy()
    for b in border_labels:
        mask_clean[mask_clean == b] = 0

    # Relabel consecutively
    mask_clean = label(mask_clean > 0, connectivity=2).astype(np.uint16)

    return mask_clean



