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
		libboost-all-dev \
		cmake \
		build-essential \
		git \
		python3-pip \
		python3-venv \
		python3-tk \
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/*

# Install ROS Packages
#RUN apt-get update && apt-get install -y \
#		ros-melodic-perception-pcl \
#		&& apt-get clean \
#		&& rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/borglab/gtsam.git
#WORKDIR /gtsam
#RUN mkdir build

# Install pip3 packages
RUN pip3 install virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install numpy scipy pyyaml matplotlib tk sympy gtsam black
RUN pip3 install plotly
RUN pip3 install pandas
RUN pip3 install kaleido
RUN useradd luckierdodge
RUN chown -R luckierdodge /opt/venv

CMD ["bash", "-c", ]
