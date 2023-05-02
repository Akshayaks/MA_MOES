# FROM nvidia/cuda:10.1-base
FROM ubuntu:20.04
# FROM nvidia/opengl
# RUN apt-get update && apt-get install python3-tk -y
# tk-dev && rm -r /var/lib/apt/lists/*
# FROM tensorflow/tensorflow:1.3.0-gpu-py3
# CMD nvidia-smi
# FROM python:3.6

#Prevent dpkg from asking for user input during installation
ARG DEBIAN_FRONTED=noninteractive

LABEL maintainer="Akshaya KS"

# Some GPG key error occurs sometimes: https://github.com/NVIDIA/nvidia-docker/issues/1632
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# install any additional dependencies
# RUN apt-get update && apt-get upgrade -y && apt-get install --no-install-recommends --no-install-suggests -y curl wget
# RUN apt-get install -q -y unzip apt-utils build-essential libssl-dev libffi-dev python3-dev python3-pip libjpeg-dev zlib1g-dev libsm6 libxext6 libxrender-dev

# set a directory for the app
WORKDIR /usr/src/MA_MOES

# copy all the files to the container
COPY . .


# install dependencies
# RUN apt-get -y install libc-dev
# RUN apt-get -y install build-essential
# RUN pip install -U pip

RUN apt-get update -y
RUN apt-get install -y python3
ENTRYPOINT ["python3"]

RUN apt-get install -y python3-pip
RUN pip3 install jax==0.2.20
RUN pip3 install jaxlib==0.1.75
RUN pip3 install kneed==0.7.0
RUN pip3 install more-itertools==8.12.0
RUN pip3 install pymoo==0.4.2.2
RUN pip3 install scikit-learn==1.0.2
# RUN pip3 install numpy==1.13.3
# RUN pip3 install matplotlib
# RUN pip3 install scipy

# RUN apt-get install python-opengl

# run the command
CMD ["run_experiments.py","--method","MOES","--test_folder","./build_prob/random_maps/","--n_agents","4","--start_positions","./start_pos_random_4_agents.npy"]