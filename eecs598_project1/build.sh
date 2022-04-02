#! /bin/bash

#Script Configuration
DOCKER_NAMESPACE=$USER
CONTAINER_NAME=eecs_jupyter
IMAGE=$DOCKER_NAMESPACE/$CONTAINER_NAME:latest

docker build -t $IMAGE .
