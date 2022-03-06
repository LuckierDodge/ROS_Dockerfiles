#! /bin/bash

DOCKER_NAMESPACE=$USER
CONTAINER_NAME=ros2_foxy_jetson
IMAGE=$DOCKER_NAMESPACE/$CONTAINER_NAME:latest

docker build -t $IMAGE .

