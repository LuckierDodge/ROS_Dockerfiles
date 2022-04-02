#! /bin/bash

# Copyright (c) 2019 Javier Peralta Saenz, Ariel Mora Jimenez.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Ryan D. Lewis

# Script Configuration
CONTAINER_USER=$USER
DOCKER_NAMESPACE=$USER
CONTAINER_NAME=eecs_jupyter
USER_ID=$UID
IMAGE=$DOCKER_NAMESPACE/$CONTAINER_NAME:latest

docker run \
	--name $CONTAINER_NAME \
	--gpus all \
	--env="CONTAINER_NAME=$CONTAINER_NAME" \
	-p 9999:9999 \
	-e JUPYTER_TOKEN=token \
	-v ~/repos/ROS_Dockerfiles/eecs598_project1/working:/app/working \
	$IMAGE
