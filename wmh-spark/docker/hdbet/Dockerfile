# CPU build — PyTorch does not publish a usable `pytorch/pytorch:*-cpu` tag for 2.0.x.
# Use official CPU wheels; `constraints.txt` stops hd-bet deps from pulling a CUDA torch.
#
# GPU host + CUDA image: use `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime`, install
# hd-bet only, and run skull_strip with `--use-gpu`.
#
FROM python:3.10-slim

# hd-bet 2.0.x requires Python >=3.10 (PyPI); the main wmh-spark app may still use 3.9 locally.
COPY constraints.txt /tmp/constraints.txt

# torch>=2.1.2 required by nnunetv2; use CPU wheels only (not pytorch/pytorch:2.0.0-cuda, which
# still exists on Docker Hub for GPU + --use-gpu).
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.4.0 torchvision==0.19.0 \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "hd-bet==2.0.0" -c /tmp/constraints.txt

# PyPI 2.0+ installs the CLI as `hd-bet` (hyphen)
ENTRYPOINT ["hd-bet"]
