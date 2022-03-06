#! /bin/bash

DOCKER_NAMESPACE=$USER
CONTAINER_NAME=turtle_bot_nav
IMAGE=$DOCKER_NAMESPACE/$CONTAINER_NAME:latest

docker build -t $IMAGE .

