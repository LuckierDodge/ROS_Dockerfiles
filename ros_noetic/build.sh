#! /bin/bash

#Script Configuration
DOCKER_NAMESPACE=$USER
CONTAINER_NAME=ros_noetic
IMAGE=$DOCKER_NAMESPACE/$CONTAINER_NAME:latest

docker build -t $IMAGE .
