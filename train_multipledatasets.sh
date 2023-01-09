#!/bin/bash

FOLDER=${1}
DATAPATH=${2}
EPOCHS=${3}
BATCH_SIZE=${4}
NORMALIZE_BACKGROUND=${5}
NUM_FILTERS=${6}
NETWORK=${7}
WANTED_CHANNEL=${8}
LEARNING_RATE=${9}
LOSS=${10}
PATIENCE=${11}
QUANTILE=${12}
PRETRAINED_WEIGHTS=${13}

if [ -z "$PRETRAINED_WEIGHTS" ]; 
then
    python model_call_inceptionunet.py ${FOLDER} ${DATAPATH} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --normalize_background ${NORMALIZE_BACKGROUND} --numFilters ${NUM_FILTERS} --network ${NETWORK} --wanted_channel ${WANTED_CHANNEL} --learning_rate ${LEARNING_RATE} --loss ${LOSS} --patience ${PATIENCE} --quantile ${QUANTILE}
else
    python model_call_inceptionunet.py ${FOLDER} ${DATAPATH} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --normalize_background ${NORMALIZE_BACKGROUND} --numFilters ${NUM_FILTERS} --network ${NETWORK} --wanted_channel ${WANTED_CHANNEL} --learning_rate ${LEARNING_RATE} --loss ${LOSS} --patience ${PATIENCE} --quantile ${QUANTILE} --pretrained_weights ${PRETRAINED_WEIGHTS}
fi
