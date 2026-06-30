FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/SOKENAR \
    SOKENAR_DATA_ROOT=/workspace/datasets \
    SOKENAR_ARTIFACT_ROOT=/workspace/artifacts

WORKDIR /workspace/SOKENAR

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs make libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN grep -v -E '^(bpy==|chumpy$)' /tmp/requirements.txt > /tmp/requirements.docker.txt \
    && python -m pip install --default-timeout=600 --retries 8 -r /tmp/requirements.docker.txt \
    && python -m pip install --default-timeout=600 --retries 8 --no-build-isolation chumpy \
    && python -m pip install huggingface_hub hf-transfer

COPY . /workspace/SOKENAR
RUN chmod +x docker/entrypoint.sh scripts/*.sh

ENTRYPOINT ["/workspace/SOKENAR/docker/entrypoint.sh"]
CMD ["smoke"]
