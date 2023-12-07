#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=$(shuf -i 15661-55661 -n 1)
# PORT=${PORT:-43424}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
