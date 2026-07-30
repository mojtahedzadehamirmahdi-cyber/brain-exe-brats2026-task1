# Training and Model-Selection Configuration

## Data

The method was developed exclusively using organizer-provided data for Task 1 of the BraTS 2026 Brain Metastases Challenge.

Following label-integrity quality control, 1,295 cases with valid segmentation masks were retained for development. The official validation cohort contained 179 unlabeled cases.

No external imaging dataset or externally pretrained segmentation weights were used.

## MRI Channels

| nnU-Net channel | MRI modality |
|---|---|
| `0000` | Post-contrast T1-weighted MRI (T1C) |
| `0001` | Pre-contrast T1-weighted MRI (T1N) |
| `0002` | T2-weighted FLAIR MRI (T2F) |
| `0003` | T2-weighted MRI (T2W) |

## Region-Aligned Targets

- Whole tumor: labels 1, 2, and 3
- Tumor core: labels 1 and 3
- Enhancing tumor: label 3
- Resection cavity: label 4

The nnU-Net region reconstruction order was `[2, 1, 3, 4]`.

## Cross-Validation and Training

A seeded five-fold split was preserved across the class-based and region-based experiments.

Five folds were trained for each of four configurations: class-based 2D, class-based 3D full resolution, region-based 2D, and region-based 3D full resolution.

Each fold was trained for 1,000 epochs using nnU-Net v2.8.1 and the standard `nnUNetTrainer`. The final checkpoint, `checkpoint_final.pth`, was used for inference.

## Final RC50 Inference System

The final submitted system used five region-based 2D folds and five region-based 3D full-resolution folds.

| Region | 3D weight | 2D weight |
|---|---:|---:|
| Whole tumor | 0.65 | 0.35 |
| Tumor core | 0.50 | 0.50 |
| Enhancing tumor | 0.75 | 0.25 |
| Resection cavity | 0.50 | 0.50 |

Every region was thresholded strictly above 0.50. Test-time augmentation was disabled. No connected-component filtering, minimum-volume removal, largest-component selection, lesion-rescue heuristic, or resection-cavity-specific postprocessing was applied.

## Hardware

Training was completed using NVIDIA A100 80-GB and H100 80-GB GPUs.

## Data and Model Availability

BraTS images, annotations, validation data, hidden-test data, and trained checkpoints are not redistributed through this repository. The frozen challenge container containing the trained checkpoints was submitted separately through the Synapse Docker registry.
