ARG BASE_IMAGE=osrf/ros:melodic-desktop

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
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/*

# Install ROS Packages
RUN apt-get update && apt-get install -y \
		ros-melodic-perception-pcl \
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/*

# COPY ./resolv.conf /etc/resolv.conf

# Install pip3 packages
#RUN pip3 install numpy scipy pyyaml matplotlib tk sympy

CMD ["bash"]
