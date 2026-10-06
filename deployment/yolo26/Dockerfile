# Export on CPU; the FP16 engine is built by the host Jetson's TensorRT.
FROM thermal-objects-train:jp6-v1
RUN PIP_EXTRA_INDEX_URL= python3 -m pip install --no-cache-dir --disable-pip-version-check \
    --index-url https://pypi.org/simple --no-deps ultralytics==8.4.144 && \
    PIP_EXTRA_INDEX_URL= python3 -m pip install --no-cache-dir --disable-pip-version-check \
    --index-url https://pypi.org/simple 'numpy==1.26.4' 'opencv-python==4.10.0.84' \
    'onnxslim==0.1.82' 'onnxruntime==1.23.2' 'polars==1.36.1'
ENV YOLO_AUTOINSTALL=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /models
