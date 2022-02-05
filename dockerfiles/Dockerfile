FROM nvidia/cuda:11.1-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    ca-certificates \
    libopenmpi-dev \
    sudo \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /tsp
WORKDIR /tsp

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \ 
    chown -R user:user /tsp
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh

# CUDA 11.0-specific steps
RUN conda install pytorch==1.9.1 torchvision cudatoolkit=11.1 -c pytorch -c conda-forge

RUN conda install numpy pandas matplotlib scikit-learn tqdm jupyter ffmpeg -c conda-forge

RUN conda install pytorch-lightning einops black flake8 isort loguru -c conda-forge 

# testing library
COPY ./tests/requirements.txt /etc/test_requirements.txt
RUN pip install --user --no-cache-dir -r /etc/test_requirements.txt

# Logger - ML flow
RUN pip install mlflow

# install self as module
COPY ./setup.py /tsp/setup.py
RUN pip install --user --no-cache-dir -e .

# project specifics
RUN pip install --user --no-cache-dir torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html && \
    pip install --user --no-cache-dir torch-geometric 

# Set the default command to python3
CMD ["python3"]