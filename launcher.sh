#!bin/sh
CONFIG=$1
WORKDIR=$2

shift 2

python train_point.py --py-config $CONFIG --work-dir $WORKDIR "$@"
