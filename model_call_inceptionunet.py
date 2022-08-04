# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Creating multiple input model that takes generators as data.

InceptionUnet modified from:
https://github.com/danielenricocahall/Keras-UNet/blob/master/UNet/createtInceptionUNet.py

Some functions based on Lippeveld et al. study:
https://github.com/saeyslab/cifconvert

Example run:
sbatch -o ./slurm-%j_output_8filter.txt 
-e ./slurm-%j_errors_8filter.txt --gres=gpu --time=100:00:00 --cpus-per-task 10 --mem=20G 
train_multipledatasets.sh /path/to/folder/ 100 22 0 8 MNC inceptionunet 5

train.sh structure:
FOLDER=${1}
DATAPATH=${2}
EPOCHS=${3}
BATCH_SIZE=${4}
NORMALIZE_BACKGROUND=${5}
NUM_FILTERS=${6}
DATASET=${7}
NETWORK=${8}
TRAIN=${9}
VAL=${10}
WHICH_RUN=${11}
WANTED_CHANNEL=${12}
LEARNING_RATE=${13}
LOSS=${14}
PATIENCE=${15}
"""

from InceptionUnet import createInceptionUnet
from generator_multipledata import generator_multipledata
from keras.layers import *
from keras.models import Model
from functions import normalize_background, normalize
from sklearn import model_selection
import argparse
import datetime
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import random

random.seed(42)

parser = argparse.ArgumentParser(description="Train DeepIFC")
parser.add_argument("folder", type=str, help="Output folder for results")
parser.add_argument("DATAPATH", type=str, help="Home path for datasets")
parser.add_argument("BASEPATH", type=str, help="Base path for training and saving results for each run of the model")
parser.add_argument("-e", "--epochs", type=int, help="Epochs", default=100)
parser.add_argument("-b", "--batch_size", type=int, help="Batch size", default=10)
parser.add_argument("--normalize_background", type=int, help="0=False or 1=True for background normalization", default=1)
parser.add_argument("--quantile", type=float, help="If normalize_background=True, which quantile is used. Default 0.6", default=0.6)
parser.add_argument("--numFilters", type=int, help="Amount of filters", default=8)
parser.add_argument("--learning_rate", type=float, help="Learning rate", default=0.0002)
parser.add_argument("--network", type=str, help="Which network to use, default inceptionunet", default="inceptionunet")
parser.add_argument("--patience", type=int, help="Patience", default=15)
parser.add_argument("--loss", type=str, help="Loss", default="binary_crossentropy")
parser.add_argument("--wanted_channel", type=int, help="Which channel to train for", default="2")
parser.add_argument("--pretrained_weights", help="Pretrained weights to import", default=None)
parser.add_argument("-ct", "--target_channels", type=int, nargs="+", help="Specify target channels as a list separated by spaces", default=[2,3,4,5,8,10,11])
args = parser.parse_args()

epochs = args.epochs 
network = args.network
folder = args.folder 
quantile = args.quantile
batch_size = args.batch_size 
learning_rate = args.learning_rate
target_channels = args.target_channels
numFilters = args.numFilters
patience = args.patience 
loss = args.loss
wanted_target_channel = args.wanted_channel
DATAPATH = args.DATAPATH
BASEPATH = args.BASEPATH
pretrained_weights = args.pretrained_weights

if pretrained_weights == None:
    pretrained_weights = None

SAVEPATH = BASEPATH + folder

########### create directory if it does not exist ##################
if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)

if not os.path.exists(SAVEPATH + "channel_{}".format(str(wanted_target_channel))):
    os.makedirs(SAVEPATH + "channel_{}".format(str(wanted_target_channel)))

if not os.path.exists(SAVEPATH + "notes"):
    os.makedirs(SAVEPATH + "notes")

if not os.path.exists(SAVEPATH + "channel_{}".format(str(wanted_target_channel))):
    os.makedirs(SAVEPATH + "channel_{}".format(str(wanted_target_channel)))

if not os.path.exists(SAVEPATH + "notes"):
    os.makedirs(SAVEPATH + "notes")

# log time
time_file = open(SAVEPATH + "notes/time_run_ch{}.txt".format(wanted_target_channel), "a")
time_file.write("Run Started: " + str(datetime.datetime.now()))

wanted_y_ch = wanted_target_channel - 1 # matrix starts at 0

if args.normalize_background == 1:
    norm_background = True
elif args.normalize_background == 0:
    norm_background = False

# fitting model 
input_shape = (128,128,3)
n_labels = 1

model = createInceptionUnet(pretrained_weights=pretrained_weights, input_shape=(128,128,3), n_labels=1,
numFilters=numFilters, output_mode="sigmoid")

optimizer = keras.optimizers.Adam(learning_rate=learning_rate) 
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mse"]) 

print("Summary of model:")
model.summary()

epochs = epochs
batch_size = batch_size

checkpoint = keras.callbacks.ModelCheckpoint(SAVEPATH + "channel_{}/{}_checkpoint_target_ch{}.h5".format(str(wanted_target_channel), str(network),
str(wanted_target_channel)),
verbose=1, monitor="val_loss",
save_best_only=True, mode="auto",
save_weights_only=True)

earlystopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
mode="min")

gen = generator_multipledata(DATAPATH, batch_size, "train", wanted_y_ch, normalized_background=norm_background, quantile=quantile)
val_gen = generator_multipledata(DATAPATH, batch_size, "val", wanted_y_ch, normalized_background=norm_background, quantile=quantile)
len_train =  # TODO
len_val =  # TODO

history = model.fit(
    gen,
    steps_per_epoch=len_train//batch_size, 
    epochs=epochs,
    validation_data=val_gen,
    use_multiprocessing=False, #using as True may lead to data duplication
    validation_steps=len_val//batch_size,
    max_queue_size=30,
    shuffle=False,
    workers=1,
    verbose=1,
    callbacks=[checkpoint, earlystopping])

print("Training model done for channel ", str(wanted_target_channel))
 
# save history
print("Saving history...")

np.save(SAVEPATH + "channel_{}/{}_accuracy_ch{}.npy".format(str(wanted_target_channel), str(network), str(wanted_target_channel)), np.array(history.history["accuracy"]))
print("Accuracy history saved.")

np.save(SAVEPATH + "channel_{}/{}_val_accuracy_ch{}.npy".format(str(wanted_target_channel), str(network), str(wanted_target_channel)), np.array(history.history["val_accuracy"]))
print("Validation accuracy history saved.")

np.save(SAVEPATH + "channel_{}/{}_loss_ch{}.npy".format(str(wanted_target_channel), str(network), str(wanted_target_channel)), np.array(history.history["loss"]))
print("Loss history saved.")

np.save(SAVEPATH + "channel_{}/{}_val_loss_ch{}.npy".format(str(wanted_target_channel), str(network), str(wanted_target_channel)), np.array(history.history["val_loss"]))
print("Validation loss history saved.")

np.save(SAVEPATH + "channel_{}/{}_mse_ch{}.npy".format(str(wanted_target_channel), str(network), str(wanted_target_channel)), np.array(history.history["mse"]))
print("MSE history saved.")

np.save(SAVEPATH + "channel_{}/{}_val_mse_ch{}.npy".format(str(wanted_target_channel), str(network), str(wanted_target_channel)), np.array(history.history["val_mse"]))
print("Val MSE history saved.")

# plot metrics
train_acc = np.array(history.history["accuracy"])
val_acc = np.array(history.history["val_accuracy"])
train_loss = np.array(history.history["loss"])
val_loss = np.array(history.history["val_loss"])
train_mse = np.array(history.history["mse"])
val_mse = np.array(history.history["val_mse"])

plt.plot(train_loss, "r", label="Training loss")
plt.plot(val_loss, "g",label="Validation loss")
plt.legend(loc="upper right")
plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Ch{}, {} filters, loss".format(str(wanted_target_channel),str(numFilters)), fontsize=5)
plt.savefig(SAVEPATH + "channel_{}/log-scale-loss_y_{}_filters_model.png".format(str(wanted_target_channel),str(numFilters)))
plt.close()

plt.plot(train_mse, "r", label="Training MSE")
plt.plot(val_mse, "g",label="Validation MSE")
plt.yscale("log")
plt.legend(loc="center right")
plt.title("Ch{}, {} filters".format(str(wanted_target_channel),str(numFilters)), fontsize=5)
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.savefig(SAVEPATH + "channel_{}/mse_{}_filters_model.png".format(str(wanted_target_channel),str(numFilters)))
plt.close()

time_file.write("Run Ended: " + str(datetime.datetime.now()))
time_file.close()