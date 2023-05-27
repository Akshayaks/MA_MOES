# FROM nvidia/cuda:11.2.0-base
FROM ubuntu:20.04
# FROM nvidia/opengl
# RUN apt-get update && apt-get install python3-tk -y
# tk-dev && rm -r /var/lib/apt/lists/*
FROM tensorflow/tensorflow:1.3.0-gpu-py3
CMD nvidia-smi
# FROM python:3.6

#Prevent dpkg from asking for user input during installation
ARG DEBIAN_FRONTED=noninteractive

LABEL maintainer="Dayi Ethan Dong"

# Some GPG key error occurs sometimes: https://github.com/NVIDIA/nvidia-docker/issues/1632
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# install any additional dependencies
# RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends --no-install-suggests -y curl wget
# RUN apt-get install -q -y unzip apt-utils build-essential libssl-dev libffi-dev python3-dev python3-pip libjpeg-dev zlib1g-dev libsm6 libxext6 libxrender-dev

# set a directory for the app
WORKDIR /usr/src/noneuclidean_search

# copy all the files to the container
COPY . .

RUN apt-get update -y
RUN apt-get install -y python3
ENTRYPOINT ["python3"]
# install dependencies
# RUN pip install cyipopt
# RUN pip3 install jax
# RUN pip3 install jaxlib
# RUN pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install jax==0.2.20
RUN pip3 install jaxlib==0.1.75
RUN pip3 install scipy
# RUN pip3 install pickle
RUN pip3 install numpy
# RUN pip3 install jaxlie
RUN pip3 install trimesh
RUN pip3 install plotly
# RUN apt-get install python-opengl

# run the command
# CMD ["python", "experiments/meshes/test_runs.py"]