# Use the Ubuntu 16.04 image as a base
FROM ubuntu:16.04

# Set up some basic environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install basic dependencies and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        python3 \
        python3-pip \
        python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Revert pip to an older version compatible with Python 3.5
RUN pip3 install --upgrade "pip<10"

# Install PyTorch 0.4
RUN pip3 install http://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision

# Set the default command to python3
CMD ["python3"]