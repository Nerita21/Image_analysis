# Image_analysis

**Overview:**

This repository contains a small pipeline for converting ND2 → TIFF, segmenting images with Cellpose, manually correcting masks in Napari, training a Cellpose model, and quantifying signal intensity (nuclear / cytoplasmic / granule-level) from two-channel FISH images (example: lncPVT1 in H1792 cells).

**Key features:**
- **Preprocessing:** convert `.nd2` → per-channel `.tiff` files.
- **Segmentation:** initial Cellpose segmentation + manual correction in Napari.
- **Training:** tools to prepare masks and fine-tune a Cellpose model (designed for noisy cytoplasm).
- **Quantification:** background subtraction, optional normalization, granule detection and summary CSV output.

**Project layout (important files):**
- `config.yaml`: project paths used by scripts.
- `src/pipeline.py`: run full pipeline or individual steps (`convert`, `segment`, `view_masks`, `train`, `measure`).
- `src/package/tiff_viewer.py`: Napari viewer helpers (`view_masks_napari`, `view_tiff_directory`).
- `src/package/intensity_measurer.py`: intensity extraction and batch helpers.
- `src/quantify_image.py`: example script that runs extraction across samples and writes CSV.

**Expected data layout** (recommended):
- Intensity TIFFs: `data/images/tiff/<sample_name>/<image_basename>_ch1.tiff` (and `_ch2.tiff` if present)
- Masks: `data/images/masks/<sample_name>/<image_basename>_ch1_masks.tiff` (and `_ch2_masks.tiff` or `_nuc_masks.tiff` for nuclear masks)

Notes: the batch tools try multiple filename patterns and both `.tiff` and `.tif` extensions to match your files.

---

## Aim of the image processing workflow

The original project focused on analyzing two-channel FISH images (.nd2 format) that visualize lncPVT1 localization in the H1792 cell line. Because H1792 cells often show high cytoplasmic background, this tailored workflow used:

- Fine-tuning Cellpose for noisy cytoplasm (use `cpsam`/`cyto` variants as a starting point).
- Manual mask correction in Napari: after generating initial masks, masks were manually corrected (this was done interactively in Napari and saved back to the masks folder).

### Running training (GPU required)

Training was performed separately (on Google Colab / other GPU hosts) using the notebook(s) in the repository. The typical workflow was:

1. Convert ND2 → TIFF and collect training images/masks into a drive folder.
2. Run the training notebook on Colab (easy GPU access) to fine-tune a Cellpose model.
3. Save the trained model to the drive and copy it back to the local `models/` or project folder as needed.

Important: training is decoupled from the local pipeline scripts because Cellpose training requires GPU resources and the notebooks used in Colab are the recommended entry point.

### Manual mask correction (Napari)

After generating initial masks with Cellpose, masks were opened in Napari for manual correction (paint/erase). Corrected masks should be saved to `data/images/masks/<sample>/` using the filename patterns described above (e.g. `<image_basename>_ch1_masks.tiff`). The pipeline's `view_masks_napari` helper is intended to streamline this step.

---

**Quick start**

1) Create environment and install dependencies (conda example):
```bash
conda env create -f environment.yml
conda activate image_analysis
pip install -r requirements.txt
```

2) Edit `config.yaml` to point `data.*` paths to your workspace (most scripts load `config.yaml`).

3) Run the pipeline (from repo root):
- Full pipeline:
```bash
python src/pipeline.py all
```
- Or run individual steps, e.g. convert only:
```bash
python src/pipeline.py convert
```
- To view masks in Napari for manual correction:
```bash
python src/pipeline.py view_masks
```

4) Quantify images (example):
```bash
python src/quantify_image.py
```
This script reads `data/images/tiff` and `data/images/masks` (by default), processes each sample subfolder, and writes `data/csv/intensity_measurements.csv`.

**Usage tips & troubleshooting**
- If masks aren't found, check the mask filename patterns in `data/images/masks/<sample>/` — common patterns are `_ch1_masks` and `_ch2_masks` (plural `masks`).
- If you get import errors when running scripts, run them from the repository root so `src`/`package` imports resolve, or add the repo root to `PYTHONPATH`.
- For consistent intensity measurements across experiments, perform background subtraction (the code does this by default) and avoid per-image aggressive normalization unless comparing within-image features only.






