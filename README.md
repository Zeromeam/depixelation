# Depixelation

A computer-vision experiment for reconstructing deliberately block-corrupted
grayscale images. The model learns to recover structure inside a pixelated region
while using the surrounding image as context.

![Reconstruction produced by the notebook](https://github.com/Zeromeam/depixelation/assets/102630502/8b87d0e4-fb11-49b4-b03d-7ce2fe56ae0f)

## Approach

The project compares residual convolutional models with a compact Deep Recursive
Laplacian Network (DRLN). Its data pipeline generates corrupted inputs and
known-pixel masks, applies augmentation, and pairs each input with its reference
image. Training uses Adam optimization with mean-squared error.

The notebook covers:

- image preprocessing, corruption, and augmentation;
- residual and DRLN model definitions;
- training and validation loss tracking;
- aligned corrupted, reconstructed, and reference examples; and
- model and prediction serialization.

## Repository structure

- `model/model.ipynb` — end-to-end experiment and recorded outputs
- `model/ops.py` — convolutional and residual building blocks
- `model/helpr_fun.py` — image preparation, pixelation, and dataset utilities
- `model/submission_serialization.py` — prediction serialization helpers

## Running the experiment

1. Create a Python environment with PyTorch, NumPy, Pillow, Matplotlib, and tqdm.
2. Configure the image dataset paths in `model/model.ipynb`.
3. Run the preprocessing, model, training, and evaluation sections in order.
4. Keep the data split and corruption seed fixed when comparing model variants.

The committed notebook outputs include loss curves and reconstruction examples
for reviewing the experiment without rerunning training. Serialized PyTorch and
pickle artifacts should be opened only in a trusted environment.

Explore the aligned restoration viewer in the
[portfolio case study](https://medoali.at/work/depixelation).
