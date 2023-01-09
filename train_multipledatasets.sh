#!/bin/bash

BASEPATH=${1}
FOLDER=${2}
DATAPATH=${3}
EPOCHS=${4}
BATCH_SIZE=${5}
NORMALIZE_BACKGROUND=${6}
NUM_FILTERS=${7}
NETWORK=${8}
WANTED_CHANNEL=${9}
LEARNING_RATE=${10}
LOSS=${11}
PATIENCE=${12}
QUANTILE=${13}
PRETRAINED_WEIGHTS=${14}

if [ -z "$PRETRAINED_WEIGHTS" ]; 
then
    python model_call_inceptionunet.py ${BASEPATH} ${FOLDER} ${DATAPATH} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --normalize_background ${NORMALIZE_BACKGROUND} --numFilters ${NUM_FILTERS} --network ${NETWORK} --wanted_channel ${WANTED_CHANNEL} --learning_rate ${LEARNING_RATE} --loss ${LOSS} --patience ${PATIENCE} --quantile ${QUANTILE}
else
    python model_call_inceptionunet.py ${BASEPATH} ${FOLDER} ${DATAPATH} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --normalize_background ${NORMALIZE_BACKGROUND} --numFilters ${NUM_FILTERS} --network ${NETWORK} --wanted_channel ${WANTED_CHANNEL} --learning_rate ${LEARNING_RATE} --loss ${LOSS} --patience ${PATIENCE} --quantile ${QUANTILE} --pretrained_weights ${PRETRAINED_WEIGHTS}
fi
