#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
# PORT=${PORT:-29824}
PORT=$(shuf -i 15661-55661 -n 1)

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
