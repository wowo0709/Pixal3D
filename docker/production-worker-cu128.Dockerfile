FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxkbcommon-x11-0 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/youngwoo/Pixal3D
CMD ["sleep", "infinity"]
