#! /bin/bash

DOCKER_NAMESPACE=$USER
CONTAINER_NAME=webots_ros2_foxy
IMAGE=$DOCKER_NAMESPACE/$CONTAINER_NAME:latest

docker build -t $IMAGE .

