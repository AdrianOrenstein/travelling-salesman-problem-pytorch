#!/usr/bin/env bash

cmp_volumes="--volume=$(pwd):/tsp/:rw"

docker run --rm -ti \
    $cmp_volumes \
    -it \
    --gpus all \
    --ipc host \
    adrianorenstein/tsp \
    ${@:-bash}