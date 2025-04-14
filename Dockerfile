FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    sudo \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m ensurepip --upgrade \
    && ln -s /opt/venv/bin/pip /usr/local/bin/pip

RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 && \
    pip install torchprofile==0.0.4

WORKDIR /workspace

CMD ["/bin/bash"]
