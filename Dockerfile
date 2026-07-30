ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="brain.exe BraTS 2026 Task 1"
LABEL org.opencontainers.image.description="Dataset702 five-fold 2D/3D RC50 ensemble"
LABEL org.opencontainers.image.version="dataset702-rc50-v1"
LABEL brainexe.dataset="Dataset702_BrainEXE_METS2026_REGIONS"
LABEL brainexe.submission_id="9774198"
LABEL brainexe.selection="RC50_FINAL_FROZEN"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV nnUNet_raw=/opt/nnunet_raw
ENV nnUNet_preprocessed=/opt/nnunet_preprocessed
ENV nnUNet_results=/opt/nnunet_results
ENV nnUNet_compile=f

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install \
    --no-cache-dir \
    nnunetv2==2.8.1 \
    nibabel==5.3.2 \
    SimpleITK==2.5.2

RUN mkdir -p \
    /app \
    /input \
    /output \
    /tmp/brainexe \
    /opt/nnunet_raw \
    /opt/nnunet_preprocessed \
    /opt/nnunet_results

ENV TMPDIR=/tmp/brainexe

COPY model/ /opt/nnunet_results/
COPY run_inference.py /app/run_inference.py

RUN test "$(find \
        /opt/nnunet_results/Dataset702_BrainEXE_METS2026_REGIONS \
        -type f \
        -name checkpoint_final.pth \
        | wc -l)" -eq 10 \
    && test "$(find \
        /opt/nnunet_results/Dataset702_BrainEXE_METS2026_REGIONS \
        -type f \
        -name dataset.json \
        | wc -l)" -eq 2 \
    && test "$(find \
        /opt/nnunet_results/Dataset702_BrainEXE_METS2026_REGIONS \
        -type f \
        -name plans.json \
        | wc -l)" -eq 2 \
    && python -m py_compile /app/run_inference.py

WORKDIR /app

ENTRYPOINT ["python", "-u", "/app/run_inference.py"]
