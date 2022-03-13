#! /bin/bash

#Script Configuration
DOCKER_NAMESPACE=$USER
CONTAINER_NAME=rob530_hw5
IMAGE=$DOCKER_NAMESPACE/$CONTAINER_NAME:latest

docker build -t $IMAGE .
