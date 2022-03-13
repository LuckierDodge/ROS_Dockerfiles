ARG BASE_IMAGE=osrf/ros:noetic-desktop

# Install dev packages
FROM $BASE_IMAGE as base
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
		tmux \
		curl \
		wget \
		vim \
		sudo \
		unzip \
		git \
		python3-tk \
		python3-pip \
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/*

# Install ROS Packages
# RUN apt-get update && apt-get install -y \
# 		&& apt-get clean \
# 		&& rm -rf /var/lib/apt/lists/*

# COPY ./resolv.conf /etc/resolv.conf

# Install pip3 packages
RUN pip3 install numpy scipy pyyaml matplotlib tk sympy

CMD ["bash"]
