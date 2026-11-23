FROM docker.io/library/python:3.12-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4
ENV NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp \
    HF_HOME=/tmp/hf HF_DEACTIVATE_ASYNC_LOAD=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
WORKDIR /opt/inference
COPY wheels.lock .
RUN python -m pip --isolated install --no-cache-dir --only-binary=:all: --require-hashes --no-deps --no-index -r wheels.lock && python -m pip check
COPY placement.py infer.py score_cases.py seven-prospective.json test_loader.py ./
USER 1000:1000
RUN --network=none python test_loader.py
ENTRYPOINT ["timeout","--kill-after=15s","20m","python","/opt/inference/infer.py"]
