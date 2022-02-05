#!/usr/bin/env bash

if [ "$DOCKER_RUNNING" == true ] 
then
    echo "Inside docker instance"
    jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
    
else
    echo "Starting up docker instance..."

    cmp_volumes="--volume=$(pwd):/tsp/:rw"

    docker run --rm -ti \
        $cmp_volumes \
        -it \
        --gpus all \
        --ipc host \
        -p 8888:8888 \
        adrianorenstein/tsp \
        jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
fi