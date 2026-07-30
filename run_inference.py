#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


DATASET_NAME = "Dataset702_BrainEXE_METS2026_REGIONS"
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"
CHECKPOINT = "checkpoint_final.pth"
FOLDS = ("0", "1", "2", "3", "4")

CONFIGURATION_3D = "3d_fullres"
CONFIGURATION_2D = "2d"

# Region-channel semantics established by Dataset702:
# channel 0 = WT, channel 1 = TC, channel 2 = ET, channel 3 = RC.
#
# nnU-Net regions_class_order = [2, 1, 3, 4].
# Later channels overwrite earlier channels.
OUTPUT_LABELS = (2, 1, 3, 4)

WEIGHTS_3D = np.asarray(
    (0.65, 0.50, 0.75, 0.50),
    dtype=np.float32,
)

WEIGHTS_2D = np.asarray(
    (0.35, 0.50, 0.25, 0.50),
    dtype=np.float32,
)

PROBABILITY_THRESHOLD = 0.50
ALLOWED_LABELS = {0, 1, 2, 3, 4}

CASE_PATTERN = re.compile(
    r"(BraTS-MET-\d{5}-\d{3}|\d{5}-\d{3})",
    flags=re.IGNORECASE,
)

MODALITY_TO_INDEX = {
    "t1c": 0,
    "t1n": 1,
    "t2f": 2,
    "t2w": 3,
}

INDEX_TO_MODALITY = {
    0: "t1c",
    1: "t1n",
    2: "t2f",
    3: "t2w",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "brain.exe BraTS 2026 Task 1 "
            "Dataset702 RC50 inference"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/input"),
        help="Challenge input directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/output"),
        help="Challenge output directory.",
    )

    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit input discovery without running inference.",
    )

    parser.add_argument(
        "--num-preprocessing-processes",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-export-processes",
        type=int,
        default=4,
    )

    return parser.parse_args()


def normalize_case_id(*candidate_texts: str) -> str:
    for text in candidate_texts:
        match = CASE_PATTERN.search(text)

        if match is None:
            continue

        case_id = match.group(1)

        if not case_id.lower().startswith("brats-met-"):
            case_id = f"BraTS-MET-{case_id}"

        prefix, identifier, timepoint = case_id.split("-")[-3:]
        return f"BraTS-MET-{identifier}-{timepoint}"

    raise ValueError(
        "Could not identify a BraTS-MET case ID from: "
        + ", ".join(candidate_texts)
    )


def identify_modality(path: Path) -> int | None:
    name = path.name.lower()

    indexed_match = re.search(
        r"_(000[0-3])\.nii\.gz$",
        name,
    )

    if indexed_match is not None:
        return int(indexed_match.group(1))

    for modality, index in MODALITY_TO_INDEX.items():
        pattern = rf"(?:^|[-_]){modality}(?=(?:[-_.]|$))"

        if re.search(pattern, name):
            return index

    return None


def collect_modalities(
    case_id: str,
    files: list[Path],
) -> dict[int, Path]:
    modalities: dict[int, Path] = {}

    for path in sorted(files):
        modality_index = identify_modality(path)

        if modality_index is None:
            continue

        if modality_index in modalities:
            raise RuntimeError(
                f"Duplicate {INDEX_TO_MODALITY[modality_index]} "
                f"files for {case_id}:\n"
                f"  {modalities[modality_index]}\n"
                f"  {path}"
            )

        modalities[modality_index] = path

    missing = sorted(set(range(4)) - set(modalities))

    if missing:
        names = [
            INDEX_TO_MODALITY[index]
            for index in missing
        ]

        raise RuntimeError(
            f"{case_id} is missing modalities: {names}"
        )

    return modalities


def discover_cases(
    input_directory: Path,
) -> dict[str, dict[int, Path]]:
    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )

    grouped_files: dict[str, list[Path]] = {}

    for path in sorted(input_directory.rglob("*.nii.gz")):
        # Ignore macOS AppleDouble/resource-fork files and
        # metadata extracted under __MACOSX directories.
        if (
            path.name.startswith("._")
            or path.name.startswith(".")
            or "__MACOSX" in path.parts
        ):
            continue

        try:
            case_id = normalize_case_id(
                path.name,
                path.parent.name,
            )
        except ValueError:
            continue

        grouped_files.setdefault(case_id, []).append(path)

    if not grouped_files:
        raise RuntimeError(
            f"No BraTS-MET NIfTI inputs were discovered in "
            f"{input_directory}"
        )

    cases: dict[str, dict[int, Path]] = {}

    for case_id, files in sorted(grouped_files.items()):
        cases[case_id] = collect_modalities(
            case_id=case_id,
            files=files,
        )

    return cases


def print_input_audit(
    cases: dict[str, dict[int, Path]],
) -> None:
    print("=" * 78)
    print("BRAIN.EXE DATASET702 RC50 INPUT AUDIT")
    print("=" * 78)
    print(f"Cases discovered: {len(cases)}")

    print()
    print("First discovered cases:")

    for case_id in list(sorted(cases))[:5]:
        print()
        print(case_id)

        for index in range(4):
            modality = INDEX_TO_MODALITY[index]
            print(f"  {modality}: {cases[case_id][index]}")

    print()
    print("Input audit: PASSED")


def print_runtime_environment() -> None:
    print()
    print("=" * 78)
    print("RUNTIME ENVIRONMENT")
    print("=" * 78)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        properties = torch.cuda.get_device_properties(0)
        memory_gib = properties.total_memory / (1024**3)
        print(f"CUDA memory: {memory_gib:.2f} GiB")

    print()
    print("Frozen RC50 fusion:")
    print("  3D: WT=0.65 TC=0.50 ET=0.75 RC=0.50")
    print("  2D: WT=0.35 TC=0.50 ET=0.25 RC=0.50")
    print("  Threshold: >0.50")


def validate_model_inventory() -> None:
    model_root = (
        Path(os.environ["nnUNet_results"])
        / DATASET_NAME
    )

    missing: list[Path] = []

    for configuration in (
        CONFIGURATION_2D,
        CONFIGURATION_3D,
    ):
        configuration_root = (
            model_root
            / (
                f"{TRAINER}__{PLANS}"
                f"__{configuration}"
            )
        )

        for metadata_name in (
            "dataset.json",
            "plans.json",
        ):
            path = configuration_root / metadata_name

            if not path.is_file():
                missing.append(path)

        for fold in FOLDS:
            path = (
                configuration_root
                / f"fold_{fold}"
                / CHECKPOINT
            )

            if not path.is_file():
                missing.append(path)

    if missing:
        formatted = "\n".join(
            f"  {path}"
            for path in missing
        )

        raise RuntimeError(
            "Container model inventory is incomplete:\n"
            f"{formatted}"
        )

    print()
    print("Model inventory: PASSED")
    print("  2D checkpoints: 5/5")
    print("  3D checkpoints: 5/5")


def stage_nnunet_inputs(
    cases: dict[str, dict[int, Path]],
    staging_directory: Path,
) -> None:
    staging_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    for case_id, modalities in sorted(cases.items()):
        for index in range(4):
            source = modalities[index]
            destination = (
                staging_directory
                / f"{case_id}_{index:04d}.nii.gz"
            )

            destination.symlink_to(source.resolve())


def run_nnunet(
    staged_input: Path,
    temporary_output: Path,
    configuration: str,
    preprocessing_processes: int,
    export_processes: int,
) -> None:
    temporary_output.mkdir(
        parents=True,
        exist_ok=False,
    )

    command = [
        "nnUNetv2_predict",
        "-i",
        str(staged_input),
        "-o",
        str(temporary_output),
        "-d",
        DATASET_NAME,
        "-c",
        configuration,
        "-tr",
        TRAINER,
        "-p",
        PLANS,
        "-chk",
        CHECKPOINT,
        "-f",
        *FOLDS,
        "-npp",
        str(preprocessing_processes),
        "-nps",
        str(export_processes),
        "--disable_tta",
        "--save_probabilities",
        "-device",
        "cuda",
    ]

    print()
    print("=" * 78)
    print(f"STARTING NNUNET {configuration.upper()} INFERENCE")
    print("=" * 78)
    print(" ".join(command))
    print()

    environment = os.environ.copy()
    environment["nnUNet_compile"] = "f"
    environment["CUDA_VISIBLE_DEVICES"] = environment.get(
        "CUDA_VISIBLE_DEVICES",
        "0",
    )

    start = time.monotonic()

    subprocess.run(
        command,
        check=True,
        env=environment,
    )

    elapsed = time.monotonic() - start

    npz_count = len(
        list(temporary_output.glob("*.npz"))
    )

    nifti_count = len(
        list(temporary_output.glob("*.nii.gz"))
    )

    print()
    print(
        f"{configuration} inference completed in "
        f"{elapsed / 60:.1f} minutes"
    )
    print(f"Probability files: {npz_count}")
    print(f"NIfTI files:       {nifti_count}")


def load_probabilities(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        if "probabilities" not in archive:
            raise RuntimeError(
                f"Missing probabilities array: {path}"
            )

        probabilities = np.asarray(
            archive["probabilities"],
            dtype=np.float32,
        )

    if probabilities.ndim != 4:
        raise RuntimeError(
            f"{path}: expected four dimensions, "
            f"found {probabilities.shape}"
        )

    if probabilities.shape[0] != 4:
        raise RuntimeError(
            f"{path}: expected four channels, "
            f"found {probabilities.shape[0]}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError(
            f"{path}: contains non-finite probabilities"
        )

    return probabilities


def geometry_matches(
    first: nib.Nifti1Image,
    second: nib.Nifti1Image,
) -> bool:
    return (
        first.shape == second.shape
        and np.allclose(
            first.affine,
            second.affine,
            rtol=0,
            atol=1e-5,
        )
        and np.allclose(
            first.header.get_zooms()[:3],
            second.header.get_zooms()[:3],
            rtol=0,
            atol=1e-5,
        )
    )


def fuse_and_export(
    cases: dict[str, dict[int, Path]],
    output_2d: Path,
    output_3d: Path,
    final_output: Path,
) -> None:
    final_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_predictions = list(
        final_output.glob("*.nii.gz")
    )

    if existing_predictions:
        raise RuntimeError(
            "Output directory already contains NIfTI predictions. "
            "Refusing to overwrite them."
        )

    start = time.monotonic()

    for number, case_id in enumerate(
        sorted(cases),
        start=1,
    ):
        probability_2d_path = (
            output_2d
            / f"{case_id}.npz"
        )

        probability_3d_path = (
            output_3d
            / f"{case_id}.npz"
        )

        nifti_2d_path = (
            output_2d
            / f"{case_id}.nii.gz"
        )

        nifti_3d_path = (
            output_3d
            / f"{case_id}.nii.gz"
        )

        for path in (
            probability_2d_path,
            probability_3d_path,
            nifti_2d_path,
            nifti_3d_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing intermediate output: {path}"
                )

        probabilities_2d = load_probabilities(
            probability_2d_path
        )

        probabilities_3d = load_probabilities(
            probability_3d_path
        )

        if probabilities_2d.shape != probabilities_3d.shape:
            raise RuntimeError(
                f"{case_id}: 2D/3D probability shape mismatch "
                f"{probabilities_2d.shape} versus "
                f"{probabilities_3d.shape}"
            )

        reference = nib.load(
            str(cases[case_id][0])
        )

        prediction_2d = nib.load(
            str(nifti_2d_path)
        )

        prediction_3d = nib.load(
            str(nifti_3d_path)
        )

        if not geometry_matches(reference, prediction_2d):
            raise RuntimeError(
                f"{case_id}: 2D prediction geometry mismatch"
            )

        if not geometry_matches(reference, prediction_3d):
            raise RuntimeError(
                f"{case_id}: 3D prediction geometry mismatch"
            )

        expected_probability_shape = (
            reference.shape[2],
            reference.shape[1],
            reference.shape[0],
        )

        if probabilities_2d.shape[1:] != expected_probability_shape:
            raise RuntimeError(
                f"{case_id}: probability/native shape mismatch "
                f"{probabilities_2d.shape[1:]} versus "
                f"{expected_probability_shape}"
            )

        segmentation_zyx = np.zeros(
            probabilities_2d.shape[1:],
            dtype=np.uint8,
        )

        for channel, output_label in enumerate(
            OUTPUT_LABELS
        ):
            fused = (
                WEIGHTS_2D[channel]
                * probabilities_2d[channel]
                + WEIGHTS_3D[channel]
                * probabilities_3d[channel]
            )

            segmentation_zyx[
                fused > PROBABILITY_THRESHOLD
            ] = output_label

        segmentation_xyz = np.transpose(
            segmentation_zyx,
            (2, 1, 0),
        )

        header = reference.header.copy()
        header.set_data_dtype(np.uint8)

        final_image = nib.Nifti1Image(
            segmentation_xyz,
            reference.affine,
            header,
        )

        qform, qcode = reference.get_qform(
            coded=True
        )

        sform, scode = reference.get_sform(
            coded=True
        )

        if qform is not None:
            final_image.set_qform(
                qform,
                int(qcode),
            )

        if sform is not None:
            final_image.set_sform(
                sform,
                int(scode),
            )

        final_path = (
            final_output
            / f"{case_id}.nii.gz"
        )

        nib.save(
            final_image,
            str(final_path),
        )

        reloaded = nib.load(str(final_path))

        if not geometry_matches(reference, reloaded):
            raise RuntimeError(
                f"{case_id}: final output geometry mismatch"
            )

        output_data = np.asanyarray(
            reloaded.dataobj
        )

        if not np.all(np.isfinite(output_data)):
            raise RuntimeError(
                f"{case_id}: final output contains "
                "non-finite values"
            )

        rounded = np.rint(output_data)

        if not np.array_equal(
            output_data,
            rounded,
        ):
            raise RuntimeError(
                f"{case_id}: final output contains "
                "non-integer values"
            )

        labels = {
            int(value)
            for value in np.unique(rounded)
        }

        invalid_labels = labels - ALLOWED_LABELS

        if invalid_labels:
            raise RuntimeError(
                f"{case_id}: invalid labels "
                f"{sorted(invalid_labels)}"
            )

        print(
            f"[{number}/{len(cases)}] "
            f"RC50 fused and verified {case_id}; "
            f"labels={sorted(labels)}",
            flush=True,
        )

    final_predictions = sorted(
        final_output.glob("*.nii.gz")
    )

    if len(final_predictions) != len(cases):
        raise RuntimeError(
            f"Expected {len(cases)} outputs, "
            f"found {len(final_predictions)}"
        )

    elapsed = time.monotonic() - start

    print()
    print("=" * 78)
    print("BRAIN.EXE DATASET702 RC50 INFERENCE: PASSED")
    print("=" * 78)
    print(f"Predictions written: {len(final_predictions)}")
    print(f"Output directory: {final_output}")
    print(f"Fusion/audit time: {elapsed / 60:.1f} minutes")


def main() -> int:
    arguments = parse_arguments()

    input_directory = arguments.input.resolve()
    output_directory = arguments.output.resolve()

    print("=" * 78)
    print("brain.exe — BraTS 2026 Task 1")
    print("Dataset702 RC50 final frozen ensemble")
    print("=" * 78)
    print(f"Input:  {input_directory}")
    print(f"Output: {output_directory}")

    validate_model_inventory()
    print_runtime_environment()

    cases = discover_cases(input_directory)
    print_input_audit(cases)

    if arguments.audit_only:
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for inference."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_start = time.monotonic()

    with tempfile.TemporaryDirectory(
        prefix="brainexe_rc50_",
        dir=os.environ.get(
            "TMPDIR",
            "/tmp",
        ),
    ) as temporary_root_string:
        temporary_root = Path(
            temporary_root_string
        )

        staged_input = (
            temporary_root
            / "nnunet_input"
        )

        output_3d = (
            temporary_root
            / "prediction_3d"
        )

        output_2d = (
            temporary_root
            / "prediction_2d"
        )

        stage_nnunet_inputs(
            cases=cases,
            staging_directory=staged_input,
        )

        # Run sequentially to remain comfortably within
        # the 24-GiB A10G GPU-memory limit.
        run_nnunet(
            staged_input=staged_input,
            temporary_output=output_3d,
            configuration=CONFIGURATION_3D,
            preprocessing_processes=(
                arguments.num_preprocessing_processes
            ),
            export_processes=(
                arguments.num_export_processes
            ),
        )

        run_nnunet(
            staged_input=staged_input,
            temporary_output=output_2d,
            configuration=CONFIGURATION_2D,
            preprocessing_processes=(
                arguments.num_preprocessing_processes
            ),
            export_processes=(
                arguments.num_export_processes
            ),
        )

        fuse_and_export(
            cases=cases,
            output_2d=output_2d,
            output_3d=output_3d,
            final_output=output_directory,
        )

    overall_elapsed = time.monotonic() - overall_start

    print()
    print("=" * 78)
    print("CONTAINER EXECUTION COMPLETE")
    print("=" * 78)
    print(
        f"Total elapsed time: "
        f"{overall_elapsed / 60:.1f} minutes"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print()
        print("=" * 78, file=sys.stderr)
        print(
            "BRAIN.EXE DATASET702 RC50 INFERENCE: FAILED",
            file=sys.stderr,
        )
        print("=" * 78, file=sys.stderr)
        print(str(error), file=sys.stderr)
        raise
