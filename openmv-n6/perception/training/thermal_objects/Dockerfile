FROM dustynv/l4t-pytorch:r36.4.0
# Keep the existing CUDA-enabled Jetson torch/torchvision wheels intact.
RUN python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple \
    numpy==1.26.4 matplotlib==3.9.2 opencv-python==4.10.0.84 \
    pyyaml==6.0.2 scipy==1.14.1 tqdm==4.66.5 pandas==2.2.3 \
    seaborn==0.13.2 py-cpuinfo==9.0.0 && \
    python3 -m pip install --no-cache-dir --no-deps --index-url https://pypi.org/simple \
    ultralytics==8.3.0 ultralytics-thop==2.0.8
ENV CUDA_MODULE_LOADING=LAZY OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib
RUN python3 -c "import torch, ultralytics; assert torch.__version__.startswith('2.4.0'); print(ultralytics.__version__)"
