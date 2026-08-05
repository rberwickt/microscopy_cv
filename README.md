# microscopy_cv

Computer vision project for **inspection microscopy** with an **AmScope camera**, focused on both:

1. **Classical image processing pipelines**, and  
2. **Deep-learning segmentation with U-Net**.

## Project Focus

This repo explores an end-to-end microscopy CV workflow:

- Camera interfacing and image capture
- Image preprocessing and enhancement
- Classical segmentation baselines (thresholding)
- U-Net-based semantic segmentation experiments
- Iterative analysis in notebooks plus script-based testing

## Repository Structure

- `image_processing/` - Core notebook workflows for preprocessing, filtering, contrast handling, and classical CV experimentation
- `unet/` - U-Net training/inference experiments for segmentation of microscopy imagery
- `dataset/` - Dataset assets used by both classical and deep-learning workflows
- `threshold.py` - Baseline threshold segmentation script
- `simplest.py`, `qt.py` - Camera preview/capture examples
- `amcam.py` - Python wrapper for AmScope camera control
- `amcam.dll`, `amcam.lib` - AmScope SDK binaries required for camera integration

## Getting Started

```bash
git clone https://github.com/rberwickt/microscopy_cv.git
cd microscopy_cv
python -m venv .venv
```

Activate environment:

- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

Install dependencies:

```bash
pip install numpy opencv-python matplotlib jupyter
```

Then:

- Run scripts directly (e.g., `python threshold.py`)
- Open notebooks in `image_processing/` and `unet/` for the main experimentation workflows

## Notes

- Camera capture depends on compatible AmScope drivers/runtime.
- The repository is organized to compare and iterate between classical CV methods and learned U-Net segmentation approaches.
