# CPU build — `pytorch/pytorch` does not publish a *-cpu tag for 2.0.x on Docker Hub; use
# the official PyTorch CPU wheel index instead. For CUDA + --use-gpu, use:
#   FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
#   RUN pip install --no-cache-dir hd-bet==2.0.0
#
# hd-bet 2.0+ requires Python >=3.10 (per PyPI); local app can stay on 3.9 for non-container use.
FROM python:3.10-slim

# Align with legacy CUDA image: torch 2.0.0, then HD-BET 2.0.0
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir hd-bet==2.0.0

ENTRYPOINT ["hd_bet"]
