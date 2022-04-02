ARG BASE_IMAGE=nvidia/cuda:11.4.2-base-ubuntu20.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/root/.mujoco/mjpro150/bin

FROM $BASE_IMAGE as base
ENV SHELL=/bin/bash
ENV TZ = US/New York

RUN mkdir /etc/localtime
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app/
#COPY requirements.txt /app/requirements.txt

# Install dev packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
		tmux \
		curl \
		wget \
		vim \
		sudo \
		unzip \
		git \
		python3-pip \
		build-essential \
		apt-utils \
		xvfb \
		python-opengl \
		cmake \
		zlib1g zlib1g-dev \
		&& apt-get clean \
		&& rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init
RUN conda install -y swig

RUN mkdir -p /root/.mujoco
WORKDIR /root/.mujoco
RUN wget https://github.com/openai/mujoco-py/archive/refs/tags/1.50.1.0.tar.gz
RUN tar -xvf 1.50.1.0.tar.gz
RUN mv mujoco-py-1.50.1.0/ mjpro150/
RUN rm 1.50.1.0.tar.gz
RUN wget https://www.roboti.us/file/mjkey.txt
#RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
#RUN echo $LD_LIBRARY_PATH

#RUN pip install -r requirements.txt
ARG LD_LIBRARY_PATH
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /root/.bashrc
ARG LD_LIBRARY_PATH
RUN /bin/bash -c "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/root/.mujoco/mjpro150/bin && pip install \
	numpy \
	torch \
	jupyterlab"
RUN /bin/bash -c "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/root/.mujoco/mjpro150/bin && pip install \
	Cython"
RUN /bin/bash -c "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/root/.mujoco/mjpro150/bin && pip install \
	box2d-py \
	box2d \
	box2d-kengz \
	gym \
	Box2D \
	gym[Box2D]"

WORKDIR /app/
RUN pip uninstall -y Box2D box2d-py
RUN git clone https://github.com/pybox2d/pybox2d
WORKDIR /app/pybox2d/
RUN python setup.py clean && python setup.py build && python setup.py install
WORKDIR /app/


CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=9999", "--allow-root", "--no-browser"]
EXPOSE 8888
