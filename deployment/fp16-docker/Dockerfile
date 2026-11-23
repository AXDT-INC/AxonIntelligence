FROM docker.io/library/python:3.12-slim-bookworm@sha256:0f086103b026048cc50cb5da25ec435c023c5241a83fc44daea593377abc61c4 AS base
ENV NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 HOME=/tmp

FROM base AS downloads
COPY wheels.lock check_cpu.py /build/
RUN python /build/check_cpu.py devices && nice -n 15 ionice -c 3 python -m pip --isolated download --no-cache-dir --no-index --only-binary=:all: \
    --no-deps --require-hashes -r /build/wheels.lock --dest /wheels

FROM base
WORKDIR /opt/convert
COPY files.py convert.py verify.py check_cpu.py selftest.py entry.sh ./
RUN --network=none --mount=type=bind,from=downloads,source=/wheels,target=/wheels \
    python check_cpu.py devices && nice -n 15 ionice -c 3 python -m pip --isolated install --no-cache-dir --no-index --only-binary=:all: /wheels/*.whl && \
    python -m pip check
USER 1000:1000
RUN --network=none python check_cpu.py && nice -n 15 ionice -c 3 python selftest.py
ENTRYPOINT ["/bin/sh", "/opt/convert/entry.sh"]
