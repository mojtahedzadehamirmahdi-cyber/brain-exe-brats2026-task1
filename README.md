# brain.exe - BraTS 2026 Task 1

Official source-code and container-configuration repository for the **brain.exe** submission to Task 1 of the BraTS 2026 Brain Metastases Challenge.

## Paper

**From Class Labels to Evaluation Regions: Region-Aligned 2D-3D nnU-Net Fusion for BraTS 2026 Brain Metastasis Segmentation**

## Method overview

The final RC50 system combines five region-based 2D nnU-Net models and five region-based 3D full-resolution nnU-Net models.

The models directly predict four overlapping evaluation regions:

- Whole tumor (WT): labels 1, 2, and 3
- Tumor core (TC): labels 1 and 3
- Enhancing tumor (ET): label 3
- Resection cavity (RC): label 4

Fold-averaged 2D and 3D probabilities are combined using region-specific weights:

| Region | 3D weight | 2D weight |
|---|---:|---:|
| WT | 0.65 | 0.35 |
| TC | 0.50 | 0.50 |
| ET | 0.75 | 0.25 |
| RC | 0.50 | 0.50 |

All regions are thresholded strictly above 0.50 and reconstructed using the class order `[2, 1, 3, 4]`.

Test-time augmentation and connected-component postprocessing are disabled.

## Repository contents

- `run_inference.py`: case discovery, prediction, RC50 fusion, reconstruction, native-geometry restoration, and output verification
- `Dockerfile`: frozen Linux/AMD64 challenge environment
- `requirements.txt`: Python package requirements
- `configs/dataset702_regions.json`: MRI channels, labels, regions, and reconstruction order
- `configs/fusion_rc50.json`: frozen RC50 inference configuration
- `docs/TRAINING.md`: training and model-selection documentation

## Input modalities

| Channel | Modality |
|---|---|
| `0000` | T1C |
| `0001` | T1N |
| `0002` | T2F |
| `0003` | T2W |

## Container contract

The container searches recursively for cases under `/input` and writes one flat NIfTI segmentation per case under `/output`.

The output audit verifies complete case coverage, unique filenames, NIfTI readability, matching native dimensions and spatial geometry, labels restricted to 0-4, and a flat output directory.

No network access is required during inference.

## Software environment

- Base image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
- PyTorch: 2.8.0
- CUDA: 12.8
- nnU-Net: 2.8.1
- nibabel: 5.3.2
- SimpleITK: 2.5.2

## Building the container

The public repository does not contain the trained checkpoints. Authorized users must place the private model bundle in a local directory named `model/` before building.

```bash
docker build --platform linux/amd64 -t brainexe-brats2026-task1:dataset702-rc50-v1 .
```

The `model/` directory is excluded from Git through `.gitignore`.

## Frozen challenge container

```text
docker.synapse.org/syn75814328/brainexe-brats2026-task1:dataset702-rc50-v1
```

Image digest:

```text
sha256:4d5fcba1bcf26024dca7f1eaa8291e3da6030158d48ed06f197d13e7321d91d0
```

Official validation submission ID: `9774198`.

## Data and checkpoint availability

BraTS imaging data, reference annotations, validation data, hidden-test data, and trained checkpoints are not redistributed through this repository.

Access to challenge data is governed by the BraTS and Synapse terms of use. The trained challenge container was submitted separately through the Synapse Docker registry.

## Team

Synapse team name: `brain.exe`
