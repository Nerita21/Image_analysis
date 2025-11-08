# Image_analysis
Overview:

This project focuses on analyzing two-channel FISH images (.nd2 format) that visualize lncPVT1 localization in the H1792 cell line.
Due to the structural features of H1792 cells, the images contain high cytoplasmic background noise, which requires specialized preprocessing and segmentation.

Preprocessing in python:
- handling nd2 files
- separating channels
- set up napari viewer
  
Training cellpose model for noisy cytoplasm:
- use cellpose cpsam model
- manual mask correction done in napari
