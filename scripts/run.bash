#!/usr/bin/env bash

if [ "$DOCKER_RUNNING" == true ] 
then
    echo "Inside docker instance, I don't know why you'd want to nest terminals?"
    exit 1
    
else
    echo "Starting up docker instance..."

    cmp_volumes="--volume=$(pwd):/tsp/:rw"

    docker run --rm -ti \
        $cmp_volumes \
        -it \
        --gpus all \
        --ipc host \
        adrianorenstein/tsp \
        /bin/bash
fi