# ValiSight Development Container
# Base: ROS 2 Humble + PyTorch + CUDA for JetPack 6.2 (L4T r36.4.0)
FROM dustynv/ros:humble-desktop-l4t-r36.4.0

ENV PIP_INDEX_URL=https://pypi.jetson-ai-lab.io/jp6/cu126
ENV PIP_TRUSTED_HOST=pypi.jetson-ai-lab.io

# The Open Robotics signing key baked into the base image expired 2025-06-01,
# which makes apt-get update fail on the ros2 repo. Upstream extended the same
# key (F42ED6FBAB17C654) to 2030-06-01, so refresh it before touching apt.
RUN curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    proj-bin proj-data libproj-dev \
    libgeos-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    numpy==1.26.1 \
    pyserial \
    pyproj \
    scipy \
    matplotlib \
    filterpy

# open3d pulls dash -> Flask -> blinker, and the apt-installed blinker 1.4 is a
# distutils package pip refuses to uninstall. --ignore-installed lets pip shadow
# it instead of erroring out.
# open3d pulls dash -> Flask -> blinker, and the apt-installed blinker 1.4 is a
# distutils package pip refuses to uninstall; --ignore-installed shadows it.
# numpy is repeated here because open3d otherwise resolves numpy 2.x, which
# breaks the cv2 bindings outright and puts scipy 1.8.0 out of its declared range.
RUN pip3 install --no-cache-dir --ignore-installed blinker open3d numpy==1.26.1 || \
    echo "WARNING: open3d install failed - install manually inside container if needed"

WORKDIR /workspace

RUN echo "source /opt/ros/humble/install/setup.bash" >> /root/.bashrc

CMD ["bash"]
