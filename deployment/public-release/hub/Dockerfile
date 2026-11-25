FROM docker.io/library/python:3.12-slim@sha256:d764629ce0ddd8c71fd371e9901efb324a95789d2315a47db7e4d27e78f1b0e9
ENV HOME=/tmp HF_HOME=/cache/hf HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /tool
COPY wheels.lock .
RUN python -m pip --isolated install --no-cache-dir --no-index --no-deps --only-binary=:all: --require-hashes -r wheels.lock && python -m pip check
USER 1000:1000
