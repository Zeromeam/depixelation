# Depixelation

A computer-vision experiment in reconstructing deliberately block-corrupted
grayscale images. The central question is visual as much as numerical: does the
model restore plausible structure inside the missing region while preserving the
known pixels around it?

![A reconstruction produced by the notebook](https://github.com/Zeromeam/depixelation/assets/102630502/8b87d0e4-fb11-49b4-b03d-7ce2fe56ae0f)

## Approach

The project progressed from residual convolutional baselines to a reduced
Deep Recursive Laplacian Network (DRLN), adapted so it could be trained on the
available hardware. The data pipeline creates corrupted inputs and known-pixel
masks; the network predicts the missing content and the training loop optimizes
mean-squared error.

The executed notebook records:

- the corruption and augmentation pipeline
- residual and DRLN model definitions
- Adam optimization and train/validation loss
- aligned corrupted, reconstructed, and reference examples
- serialized model and prediction artifacts used during the experiment

The implementation lives in model/model.ipynb, with supporting operations in
model/ops.py and model/helpr_fun.py.

## Evidence

The notebook output is the source of truth for the published visual examples and
loss curve. The portfolio case study deliberately places corrupted,
reconstructed, and reference images in the same coordinate system so the missing
region can be inspected directly instead of relying on one headline metric.

## Reproduction

This repository preserves the experiment, but it is not packaged as a one-command
training project. To reproduce it:

1. Create a Python environment with PyTorch, NumPy, Pillow, Matplotlib, and tqdm.
2. Replace the machine-specific dataset paths near the start of model/model.ipynb.
3. Run the data, model, evaluation, and training sections in order.
4. Keep the train/validation split and corruption seed fixed when comparing runs.

The model directory contains historical serialized PyTorch and pickle artifacts.
Only load pickle-based files in a trusted environment.

## Limitations

- The source image dataset is not redistributed here.
- The notebook reflects an exploratory research workflow and retains local paths.
- Original pixel arrays were not retained reliably enough to publish a new
  residual-error map; the case study therefore uses only verified notebook output.
- A reconstruction can look plausible while still differing from the unknown
  original. The reference view remains essential.

Inspect the aligned restoration viewer in the
[portfolio case study](https://medoali.at/work/depixelation).
