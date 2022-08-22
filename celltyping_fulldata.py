#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create image means, roc_auc_curve, scatterplots and other plots of hdf data. Also includes static and interactive UMAPs for visualisation.

Partly based on Lippeveld et al. 2020 study:
https://github.com/saeyslab/cifconvert
"""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
from array import *
from doctest import ELLIPSIS_MARKER
from bokeh.io import output_file, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import Viridis
from bokeh.plotting import figure, show, output_notebook,output_file
from bokeh.transform import linear_cmap,factor_cmap
from InceptionUnet import createInceptionUnet
from io import BytesIO
from keras.layers import *
import itertools
from keras.models import Model
import matplotlib
from functions import normalize_background, normalize, func, percentage, naturalsorting
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import auc, roc_curve
from target_generator import combined_generator
from umap_generator import umap_generator
import argparse
import base64
import h5py
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
import umap

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
random.seed(42)

parser = argparse.ArgumentParser("Create plots")
parser.add_argument("--dataset", type=str, help="Which dataset to use")
parser.add_argument("--hdf_file", type=str, help="Which file to use for means")
parser.add_argument("--hdf_file_path", type=str, help="Which filepath to use for file for means", default="/path/to/folder/")
parser.add_argument("--means_available", type=int, help="If means for images have already been calculated, no=0 or yes=1", default=1)
parser.add_argument("--path_for_means", type=str, help="Path for saving or importing means", default="/path/to/folder/")
parser.add_argument("--folder", type=str, help="Save folder")
parser.add_argument("--color_mapping_list", nargs="+", type=str, help="List which color mappings for interactive umap", default=("CD45", "CD3", "CD14", "7AAD", "CD19", "CD8", "CD56"))
parser.add_argument("--BASEPATH", type=str, default="/path/to/folder/")
parser.add_argument("--normalize_background", type=int, help="0=False or 1=True for background normalization", default=1)
parser.add_argument("--quantile", type=float, help="If normalize_background=True, which quantile is used. Default 0.6", default=0.6)
parser.add_argument("--numFilters", type=int, help="Amount of filters", default=8)
parser.add_argument("--threshold", type=float, help="For cell typing", default=0.02)
parser.add_argument("--loss", type=str, help="Which loss to compile model with", default="binary_crossentropy")
parser.add_argument("--batch_size", type=int, help="For data generators", default=10)
parser.add_argument("--network", type=str, help="Which network is in use", default="inceptionunet")
parser.add_argument("--file_prefix", type=str, help="Which prefix for files", default="umap")
parser.add_argument("--wanted_layer", type=int, help="Bottom of network. 155 for long inceptionunet", default=155)
parser.add_argument("-nn", "--nearest_neighbor", type=int, help="Nearest neighbor, default 30", default=30)
parser.add_argument("-ct", "--target_channels", type=int, nargs="+", help="Specify target channels as a list separated by spaces", default=[2,3,4,5,8,10,11])
parser.add_argument("--data_point_amount", type=int, help="How many images in interactive umap", default=3000)
parser.add_argument("--low_quantile", type=float, help="For UMAP visualisation color scheme, lowest value for negative marker", default=0.1)
parser.add_argument("--high_quantile", type=float, help="For UMAP visualisation color scheme, highest value for positive marker", default=0.9)
args = parser.parse_args()

# import args
BASEPATH = args.BASEPATH
batch_size = args.batch_size
color_mapping_list = args.color_mapping_list
data_point_amount = args.data_point_amount #for interactive umap features 
dataset = args.dataset
file_prefix = args.file_prefix
folder = args.folder
hdf_file = args.hdf_file
hdf_file_path = args.hdf_file_path
high_quantile = args.high_quantile
loss = args.loss
low_quantile = args.low_quantile
means_available = args.means_available
n_neighbors = args.nearest_neighbor
network = args.network
numFilters = args.numFilters 
path_for_means = args.path_for_means
quantile = args.quantile 
target_channels = args.target_channels
threshold = args.threshold
wanted_layer = args.wanted_layer
if args.normalize_background == 1:
    norm_background = True
elif args.normalize_background == 0:
    norm_background = False

SAVEPATH = BASEPATH + folder + "/"

# mandatory parameters for DeepIFC
channel_name_list = ["CD45","CD3","CD14","7AAD","CD19","CD8","CD56"]
target_chans = [2,3,4,5,8,10,11]
channel_folders = ["channel_2","channel_3","channel_4","channel_5","channel_8","channel_10","channel_11"]
cell_type_list = ["Cytotoxic T Cell", "NK", "NKT", "B Cell", "T Cell", "Monocyte", "Damaged/Dead"]
threshold_7AAD = 0.007
threshold_CD19 = 0.012 
threshold_CD8 = 0.006
threshold_CD56 = 0.006

y=128
x=128

if dataset == "TESTSET":
    hdf_file = "" # TODO add dataset path
    hdf_file_path= ""
    IMAGE_PATH = hdf_file_path + hdf_file
    CROPPED_IMAGE_PATH = IMAGE_PATH # for interactive umap, which cannot fit large amounts of images
else:
    print("No dataset chosen")

"""
FUNCTIONS
"""

if not os.path.exists(SAVEPATH):
    os.makedirs(SAVEPATH)

def cell_gating_relaxed(x, threshold=0.01,threshold_7AAD=0.008,threshold_CD19=0.012,threshold_CD8=0.006,threshold_CD56=0.006): 
    """
    Permissible cell gating strategy. Dead or damaged cells can fall under multiple categories in addition to "Damaged/Dead".
    """
    # if over threshold for 7AAD
    if x["7AAD"] >= threshold_7AAD:
        if x["CD45"] >= threshold:
            if x["CD3"] >= threshold: 
                if x["CD8"] >= threshold_CD8 and x["CD56"] < threshold_CD56:
                    return ["WBC", "Cytotoxic T Cell", "T Cell","Damaged/Dead"] 

                elif x["CD56"] >= threshold_CD56: 
                    return ["WBC", "NKT", "Damaged/Dead", "T Cell"] 

                else:
                    return ["WBC", "T Cell","Damaged/Dead"]
            else:
                if x["CD56"] >= threshold_CD56:
                    return ["WBC", "NK","Damaged/Dead"]

                elif x["CD19"] >= threshold_CD19: 
                    return ["WBC", "B Cell","Damaged/Dead"]

                elif x["CD14"] >= threshold:
                    return ["WBC", "Monocyte","Damaged/Dead"]

                else:
                    return ["WBC","Damaged/Dead"]

    # if under threshold for 7AAD
    elif x["7AAD"] < threshold_7AAD:
        if x["CD45"] >= threshold:
            if x["CD3"] >= threshold: 
                if x["CD8"] >= threshold_CD8 and x["CD56"] < threshold_CD56:
                    return ["WBC", "Cytotoxic T Cell", "T Cell"] 

                elif x["CD56"] >= threshold_CD56: 
                    return ["WBC", "NKT", "T Cell"]

                else:
                    return ["WBC", "T Cell"]
            else:
                if x["CD56"] >= threshold_CD56:
                    return ["WBC", "NK"]

                elif x["CD19"] >= threshold_CD19:
                    return ["WBC", "B Cell"]

                elif x["CD14"] >= threshold:
                    return ["WBC", "Monocyte"]

                else:
                    return ["WBC"]
    # if cell doesn't fit any of above categories
    return ["Unknown"]

def cell_gating_strict(x, threshold=0.01, threshold_7AAD=0.008,threshold_CD19=0.012,threshold_CD8=0.006,threshold_CD56=0.006): 

    """
    Strict cell gating strategy. Dead or damaged cells only fall under one category.
    """
    if x["7AAD"]>= threshold_7AAD:
        return ["Damaged/Dead"]

    elif x["CD45"]>=threshold:
        if x["CD3"]>=threshold:
            if x["CD8"] >= threshold_CD8 and x["CD56"] < threshold_CD56:
                return["Cytotoxic T Cell"]

            elif x["CD56"] >= threshold_CD56: 
                    return ["NKT"]
            else:
                return ["T Cell"]
        else:
            if x["CD56"] >= threshold_CD56:
                    return ["NK"]
            elif x["CD19"] >= threshold_CD19:
                    return ["B Cell"]
            elif x["CD14"] >= threshold:
                    return ["Monocyte"]
            else:
                return ["WBC"]
    return ["Unknown"]

def calculate_means(hdf_file="TESTSET.h5", hdf_file_path="/path/to/folder/", path_for_means=path_for_means):
    with h5py.File(hdf_file_path + hdf_file, "r") as f:
        file_length = len(f["channel_1/images"])
        print("Length of test file: ", str(file_length), str(hdf_file))
    
    print("Starting to process target images, saving to ", path_for_means)
    for channel in target_chans: 
        if channel == 2:
            wanted_y_ch = 0
            model = model1
        elif channel == 3:
            wanted_y_ch = 1
            model = model2
        elif channel == 4:
            wanted_y_ch = 2
            model = model3
        elif channel == 5:
            wanted_y_ch = 3
            model = model4
        elif channel == 8:
            wanted_y_ch = 4
            model = model5
        elif channel == 10:
            wanted_y_ch = 5
            model = model6
        elif channel == 11:
            wanted_y_ch = 6
            model = model7

        gen = combined_generator(hdf_file_path + hdf_file, batch_size)
        print("Processing file ", str(hdf_file_path + hdf_file))
        print("Wanted y ch for indexing batch", wanted_y_ch, ". Processing channel ", channel)
        
        batch_number = 0 
        with open(BASEPATH + path_for_means + "/{}_dataset_ch{}_channelwisemeans_normalized_target.txt".format(str(hdf_file), str(channel)), "ab") as targetsave, open(BASEPATH + path_for_means + "/{}_dataset_ch{}_channelwisemeans_normalized_pred.txt".format(str(hdf_file),str(channel)), "ab") as predsave:
            for batch in gen: 
                if batch_number >= (file_length / batch_size):
                    break

                means_target = []
                for array in batch[1][:,:,:,wanted_y_ch]: 
                    if norm_background == True and np.max(array)>0: # because some images are zero arrays in RBC data
                        norm_array = normalize_background(array, quantile)
                    mean_t = np.mean(norm_array)
                    means_target.append(mean_t)

                np.savetxt(targetsave, means_target, delimiter="\t")

                means_pred = []
                for array in batch[0]: #(128,128,3)
                    array = array[np.newaxis, :, :, :] #(1,128,128,3)
                    predicted = model.predict(array) #(1,128,128,1)
                    predicted_norm = predicted[0,:,:,0]
                    mean_p = np.mean(predicted_norm)
                    means_pred.append(mean_p)

                np.savetxt(predsave, means_pred, delimiter="\t")
                batch_number += 1


def create_histograms(bins, CD45_pred, CD3_pred, CD14_pred, pred_7AAD, CD19_pred, CD8_pred, CD56_pred,
    CD45_gt, CD3_gt, CD14_gt, gt_7AAD, CD19_gt, CD8_gt, CD56_gt):
    # predictions
    plt.hist(CD45_pred, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram for ch2 pred")
    plt.savefig(SAVEPATH + "ch2_histogram_preds.png")
    plt.close()

    plt.hist(CD3_pred, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch3 pred")
    plt.savefig(SAVEPATH + "ch3_histogram_preds.png")
    plt.close()

    plt.hist(CD14_pred, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch4 pred")
    plt.savefig(SAVEPATH + "ch4_histogram_preds.png")
    plt.close()

    plt.hist(pred_7AAD, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch5 pred")
    plt.savefig(SAVEPATH + "ch5_histogram_preds.png")
    plt.close()

    plt.hist(CD19_pred, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch8 pred")
    plt.savefig(SAVEPATH + "ch8_histogram_preds.png")
    plt.close()

    plt.hist(CD8_pred, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch10 pred")
    plt.savefig(SAVEPATH + "ch10_histogram_preds.png")
    plt.close()

    plt.hist(CD56_pred, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch11 pred")
    plt.savefig(SAVEPATH + "ch11_histogram_preds.png")
    plt.close()

    # targets
    plt.hist(CD45_gt, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch2 target")
    plt.savefig(SAVEPATH + "ch2_histogram_targets.png")
    plt.close()

    plt.hist(CD3_gt, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch3 target")
    plt.savefig(SAVEPATH + "ch3_histogram_targets.png")
    plt.close()

    plt.hist(CD14_gt, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch4 target")
    plt.savefig(SAVEPATH + "ch4_histogram_targets.png")
    plt.close()

    plt.hist(gt_7AAD, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch5 target")
    plt.savefig(SAVEPATH + "ch5_histogram_targets.png")
    plt.close()

    plt.hist(CD19_gt, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram ch8 target")
    plt.savefig(SAVEPATH + "ch8_histogram_targets.png")
    plt.close()

    plt.hist(CD8_gt, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch10 target")
    plt.savefig(SAVEPATH + "ch10_histogram_targets.png")
    plt.close()

    plt.hist(CD56_gt, bins=bins)
    plt.xlabel("Mean value")
    plt.ylabel("Amount of cells")
    plt.title("Histogram, ch11 target")
    plt.savefig(SAVEPATH + "ch11_histogram_targets.png")
    plt.close()

def create_scatterplots(alpha, CD45_pred, CD3_pred, CD14_pred, pred_7AAD, CD19_pred, CD8_pred, CD56_pred,
    CD45_gt, CD3_gt, CD14_gt, gt_7AAD, CD19_gt, CD8_gt, CD56_gt):
    plt.scatter(CD45_gt,CD45_pred,s=5,alpha=alpha)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.axhline(y=0.01, color='gray', linestyle='--')
    plt.axvline(x=0.01, color='gray', linestyle='--')
    plt.title("Scatterplot, ch2 CD45")
    plt.savefig(SAVEPATH + "Scatterplot_ch2_CD45.png")
    plt.close()

    plt.scatter(CD3_gt,CD3_pred,s=5,alpha=alpha)
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.axhline(y=0.01, color='gray', linestyle='--')
    plt.axvline(x=0.01, color='gray', linestyle='--')
    plt.title("Scatterplot, ch3 CD3")
    plt.savefig(SAVEPATH + "Scatterplot_ch3_CD3.png")
    plt.close()

    plt.scatter(CD14_gt,CD14_pred,s=5,alpha=alpha) 
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.axhline(y=0.01, color='gray', linestyle='--')
    plt.axvline(x=0.01, color='gray', linestyle='--')
    plt.title("Scatterplot, ch4 CD14")
    plt.savefig(SAVEPATH + "Scatterplot_ch4_CD14.png")
    plt.close()

    plt.scatter(gt_7AAD,pred_7AAD,s=5,alpha=alpha) 
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.axhline(y=0.01, color='gray', linestyle='--')
    plt.axvline(x=0.01, color='gray', linestyle='--')
    plt.title("Scatterplot for, ch5 7AAD")
    plt.savefig(SAVEPATH + "Scatterplot_ch5_7AAD.png")
    plt.close()

    plt.scatter(CD19_gt,CD19_pred,s=5,alpha=0.5) 
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.axhline(y=0.01, color='gray', linestyle='--')
    plt.axvline(x=0.01, color='gray', linestyle='--')
    plt.title("Scatterplot, ch8 CD19")
    plt.savefig(SAVEPATH + "Scatterplot_ch8_CD19.png")
    plt.close()

    plt.scatter(CD8_gt,CD8_pred,s=5,alpha=alpha) 
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.axhline(y=0.01, color='gray', linestyle='--')
    plt.axvline(x=0.01, color='gray', linestyle='--')
    plt.title("Scatterplot, ch10 CD8")
    plt.savefig(SAVEPATH + "Scatterplot_ch10_CD8.png")
    plt.close()

    plt.scatter(CD56_gt,CD56_pred,s=5,alpha=alpha) 
    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.axhline(y=0.01, color='gray', linestyle='--')
    plt.axvline(x=0.01, color='gray', linestyle='--')
    plt.title("Scatterplot, ch11 CD56")
    plt.savefig(SAVEPATH + "Scatterplot_ch11_CD56.png")
    plt.close()

def create_roc_auc_curve(threshold, dataset, target_chans, CD45_pred, CD3_pred, CD14_pred, pred_7AAD, CD19_pred, CD8_pred, CD56_pred,
    CD45_gt, CD3_gt, CD14_gt, gt_7AAD, CD19_gt, CD8_gt, CD56_gt): 
    # ROC / AUC
    plt.figure(1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.plot([0, 1], [0, 1], 'k--')
    for i, channel_name in enumerate(target_chans):
        with open(SAVEPATH + "channel{}_binaryclassifications_targets.csv".format(str(channel_name)), "w+") as outfile:
            if channel_name == 2:
                true = CD45_gt
                pred = CD45_pred
            if channel_name == 3:
                true = CD3_gt
                pred = CD3_pred
            if channel_name == 4:
                true = CD14_gt
                pred = CD14_pred
            if channel_name == 5:
                true = gt_7AAD
                pred = pred_7AAD
            if channel_name == 8:
                true = CD19_gt
                pred = CD19_pred
            if channel_name == 10:
                true = CD8_gt
                pred = CD8_pred
            if channel_name == 11:
                true = CD56_gt
                pred = CD56_pred

            print("Processing channel ", channel_name)
            binary_ground_truth = []

            for j in true:
                if j >= threshold:
                    binary_ground_truth.append(1)
                else:
                    binary_ground_truth.append(0)

            fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.array(binary_ground_truth), np.array(pred))
            auc_score = auc(fpr_keras, tpr_keras)
            print(channel_name_list[i])
            plt.plot(fpr_keras, tpr_keras, label="AUC={:.3f},{}".format(auc_score,str(channel_name_list[i])))
            np.savetxt(outfile, binary_ground_truth, delimiter="\t")

    plt.ylim(ymax=1, ymin=0)
    plt.xlim(xmax=1, xmin=0)
    plt.xlabel("False positive rate", fontsize=10)
    plt.ylabel("True positive rate", fontsize=10)
    plt.title("ROC curve")
    plt.legend(loc="best")
    plt.savefig(SAVEPATH + "AUC_ROC_allmarkers_{}.png".format(str(dataset)))
    plt.close()

"""
CALLING FUNCTIONS
"""
if means_available == 0:
    # calculate means for fluorescent images
    y = 128
    x = 128

    # channels 6 and 7 are empty
    model1 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_2/inceptionunet_checkpoint_target_ch2.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model2 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_3/inceptionunet_checkpoint_target_ch3.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model3 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_4/inceptionunet_checkpoint_target_ch4.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model4 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_5/inceptionunet_checkpoint_target_ch5.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model5 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_8/inceptionunet_checkpoint_target_ch8.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model6 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_10/inceptionunet_checkpoint_target_ch10.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model7 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_11/inceptionunet_checkpoint_target_ch11.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")

    # compile models for different target channels
    model1.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model2.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model3.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model4.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model5.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model6.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model7.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])

    # calculate means for file
    print("Saving means in ", BASEPATH, path_for_means)
    print("Means for ", hdf_file)
    calculate_means(hdf_file=hdf_file, hdf_file_path=hdf_file_path, path_for_means=path_for_means)

elif means_available == 1:
    # create plots from means
    # 1. import csv files of means and initialize variables, if means are available

    print("Means available")
    CD45_target = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch2_channelwisemeans_normalized_target.txt".format(str(hdf_file)), sep="\t", header=None)
    CD3_target = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch3_channelwisemeans_normalized_target.txt".format(str(hdf_file)), sep="\t", header=None)
    CD14_target = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch4_channelwisemeans_normalized_target.txt".format(str(hdf_file)), sep="\t", header=None)
    target_7AAD = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch5_channelwisemeans_normalized_target.txt".format(str(hdf_file)), sep="\t", header=None)
    CD19_target = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch8_channelwisemeans_normalized_target.txt".format(str(hdf_file)), sep="\t", header=None)
    CD8_target = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch10_channelwisemeans_normalized_target.txt".format(str(hdf_file)), sep="\t", header=None)
    CD56_target = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch11_channelwisemeans_normalized_target.txt".format(str(hdf_file)), sep="\t", header=None)

    CD45_prediction = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch2_channelwisemeans_normalized_pred.txt".format(str(hdf_file)), sep="\t", header=None)
    CD3_prediction = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch3_channelwisemeans_normalized_pred.txt".format(str(hdf_file)), sep="\t", header=None)
    CD14_prediction = pd.read_csv(BASEPATH + path_for_means +  "/{}_dataset_ch4_channelwisemeans_normalized_pred.txt".format(str(hdf_file)), sep="\t", header=None)
    prediction_7AAD = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch5_channelwisemeans_normalized_pred.txt".format(str(hdf_file)), sep="\t", header=None)
    CD19_prediction = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch8_channelwisemeans_normalized_pred.txt".format(str(hdf_file)), sep="\t", header=None)
    CD8_prediction = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch10_channelwisemeans_normalized_pred.txt".format(str(hdf_file)), sep="\t", header=None)
    CD56_prediction = pd.read_csv(BASEPATH + path_for_means + "/{}_dataset_ch11_channelwisemeans_normalized_pred.txt".format(str(hdf_file)), sep="\t", header=None)

    # fill NaN values and replace with 0
    CD45_target = CD45_target.fillna(0)
    CD3_target = CD3_target.fillna(0)
    CD14_target = CD14_target.fillna(0)
    target_7AAD = target_7AAD.fillna(0)
    CD19_target = CD19_target.fillna(0)
    CD8_target = CD8_target.fillna(0)
    CD56_target = CD56_target.fillna(0)
    CD45_prediction = CD45_prediction.fillna(0)
    CD3_prediction = CD3_prediction.fillna(0)
    CD14_prediction = CD14_prediction.fillna(0)
    prediction_7AAD = prediction_7AAD.fillna(0)
    CD19_prediction = CD19_prediction.fillna(0)
    CD8_prediction = CD8_prediction.fillna(0)
    CD56_prediction = CD56_prediction.fillna(0)

    CD45_target = CD45_target.values.tolist()
    CD3_target = CD3_target.values.tolist()
    CD14_target = CD14_target.values.tolist()
    target_7AAD = target_7AAD.values.tolist()
    CD19_target = CD19_target.values.tolist()
    CD8_target = CD8_target.values.tolist()
    CD56_target = CD56_target.values.tolist()
    CD45_prediction = CD45_prediction.values.tolist()
    CD3_prediction = CD3_prediction.values.tolist()
    CD14_prediction = CD14_prediction.values.tolist()
    prediction_7AAD = prediction_7AAD.values.tolist()
    CD19_prediction = CD19_prediction.values.tolist()
    CD8_prediction = CD8_prediction.values.tolist()
    CD56_prediction = CD56_prediction.values.tolist()

    # unpack list of lists 
    CD45_target = list(itertools.chain(*CD45_target))
    CD3_target = list(itertools.chain(*CD3_target))
    CD14_target = list(itertools.chain(*CD14_target))
    target_7AAD = list(itertools.chain(*target_7AAD))
    CD19_target = list(itertools.chain(*CD19_target))
    CD8_target = list(itertools.chain(*CD8_target))
    CD56_target = list(itertools.chain(*CD56_target))
    CD45_prediction = list(itertools.chain(*CD45_prediction))
    CD3_prediction = list(itertools.chain(*CD3_prediction))
    CD14_prediction = list(itertools.chain(*CD14_prediction))
    prediction_7AAD = list(itertools.chain(*prediction_7AAD))
    CD19_prediction = list(itertools.chain(*CD19_prediction))
    CD8_prediction = list(itertools.chain(*CD8_prediction))
    CD56_prediction = list(itertools.chain(*CD56_prediction))

    # more strict heatmap classification
    cells_target_strict = pd.DataFrame({"CD45":CD45_target,"CD3":CD3_target,"CD14":CD14_target,"7AAD":target_7AAD,"CD19":CD19_target,"CD8":CD8_target,"CD56":CD56_target})
    cells_pred_strict = pd.DataFrame({"CD45":CD45_prediction,"CD3":CD3_prediction,"CD14":CD14_prediction,"7AAD":prediction_7AAD,"CD19":CD19_prediction,"CD8":CD8_prediction,"CD56":CD56_prediction})
    cells_target_strict["Target cell type"] = cells_target_strict.apply(cell_gating_strict, args=(threshold,threshold_7AAD,threshold_CD19,threshold_CD8,threshold_CD56), axis=1)
    cells_pred_strict["Predicted cell type"] = cells_pred_strict.apply(cell_gating_strict, args=(threshold,threshold_7AAD,threshold_CD19,threshold_CD8,threshold_CD56), axis=1)

    cells_target_strict["Target cell type"] = cells_target_strict["Target cell type"].values.tolist()
    cells_pred_strict["Predicted cell type"] = cells_pred_strict["Predicted cell type"].values.tolist()
    cells_target_strict["Target cell type"] = list(itertools.chain(*cells_target_strict["Target cell type"]))
    cells_pred_strict["Predicted cell type"] = list(itertools.chain(*cells_pred_strict["Predicted cell type"]))
    
    cells_target_strict.to_csv(SAVEPATH+"cells_target_counts_strictclassification.csv", sep="\t", header=False, index=False)
    cells_pred_strict.to_csv(SAVEPATH+"cells_pred_counts_strictclassification.csv", sep="\t", header=False, index=False)
        
    # create heatmaps for strict classification
    res = pd.crosstab(cells_pred_strict["Predicted cell type"], cells_target_strict["Target cell type"], dropna=False, normalize="index")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    res = res.reindex(index=["B Cell","Cytotoxic T Cell", "Damaged/Dead", "Monocyte", "NK", "NKT", "T Cell","Unknown","WBC"],columns=["B Cell","Cytotoxic T Cell", "Damaged/Dead", "Monocyte", "NK", "NKT", "T Cell","Unknown","WBC"],fill_value=0)

    hm = sns.heatmap(res, annot=True, fmt=".2f", square=True)
    plt.tight_layout()
    plt.xlabel("Target cell types", fontsize=10) 
    plt.ylabel("Predicted cell types", fontsize=10) 
    hm.figure.savefig(SAVEPATH + "heatmap_allchannels_rownorm.png")
    plt.close()

    # create heatmaps for counts 
    res = pd.crosstab(cells_pred_strict["Predicted cell type"], cells_target_strict["Target cell type"], dropna=False)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    res = res.reindex(index=["B Cell","Cytotoxic T Cell", "Damaged/Dead", "Monocyte", "NK", "NKT", "T Cell","Unknown","WBC"],columns=["B Cell","Cytotoxic T Cell", "Damaged/Dead", "Monocyte", "NK", "NKT", "T Cell","Unknown","WBC"],fill_value=0)

    hm = sns.heatmap(res, annot=True, fmt=".2f", annot_kws={"size":8}, square=True)
    plt.tight_layout()
    plt.xlabel("Target cell types", fontsize=10) 
    plt.ylabel("Predicted cell types", fontsize=10) 
    hm.figure.savefig(SAVEPATH + "heatmap_allchannels_counts.png")
    plt.close()

    # more relaxed heatmap classification, 7AAD adjusted for gating (for individual datasets)
    cells_target_relaxed = pd.DataFrame({"CD45":CD45_target,"CD3":CD3_target,"CD14":CD14_target,"7AAD":target_7AAD,"CD19":CD19_target,"CD8":CD8_target,"CD56":CD56_target})
    cells_pred_relaxed = pd.DataFrame({"CD45":CD45_prediction,"CD3":CD3_prediction,"CD14":CD14_prediction,"7AAD":prediction_7AAD,"CD19":CD19_prediction,"CD8":CD8_prediction,"CD56":CD56_prediction})
    cells_target_relaxed["Target cell type"] = cells_target_relaxed.apply(cell_gating_relaxed, args=(threshold,threshold_7AAD,threshold_CD19,threshold_CD8,threshold_CD56), axis=1)
    cells_pred_relaxed["Predicted cell type"] = cells_pred_relaxed.apply(cell_gating_relaxed, args=(threshold,threshold_7AAD,threshold_CD19,threshold_CD8,threshold_CD56), axis=1)
    cells_target_relaxed["Target cell type"].to_csv(SAVEPATH+"cells_target_counts_relaxedclassification.csv", sep="\t", header=False, index=False)
    cells_pred_relaxed["Predicted cell type"].to_csv(SAVEPATH+"cells_pred_counts_relaxedclassification.csv", sep="\t", header=False, index=False)

    # histograms, scatterplots and roc auc curve
    create_histograms(500, CD45_prediction, CD3_prediction, CD14_prediction, prediction_7AAD, CD19_prediction, CD8_prediction, CD56_prediction, 
    CD45_target, CD3_target, CD14_target, target_7AAD, CD19_target, CD8_target, CD56_target)
    create_scatterplots(0.3, CD45_prediction, CD3_prediction, CD14_prediction, prediction_7AAD, CD19_prediction, CD8_prediction, CD56_prediction, 
    CD45_target, CD3_target, CD14_target, target_7AAD, CD19_target, CD8_target, CD56_target)
    create_roc_auc_curve(0.01, dataset, target_chans, CD45_prediction, CD3_prediction, CD14_prediction, prediction_7AAD, CD19_prediction, CD8_prediction, CD56_prediction, 
    CD45_target, CD3_target, CD14_target, target_7AAD, CD19_target, CD8_target, CD56_target)

    # Creating percentages of all combined models for relaxed gating
    for i, celltype in enumerate(cell_type_list):
        print(celltype)
        print(cell_type_list[i])
        predfile = pd.read_csv(SAVEPATH + "cells_pred_counts_relaxedclassification.csv", sep="\t", header=None)
        pred_occurrences = []
        for index, pred in predfile.iterrows():
            if cell_type_list[i] in pred.values[0]:
                pred_occurrences.append(1)
        if len(pred_occurrences) > 0:
            print("Percentage ", percentage(len(pred_occurrences),len(predfile)))
        else:
            print("None predicted")
            continue

        combined_results = {"Percentage":percentage(len(pred_occurrences),len(predfile)),"pred_occurrences":len(pred_occurrences),"Length of file":len(predfile)}
        with open(SAVEPATH+"percentages_celltype_{}_out_of_Allgating.txt".format(celltype),"w") as f:
            print(combined_results,file=f)

    ###########################################################
    ########### IMPORT WEIGHTS, SET UP MODEL ##################
    ###########################################################

    y = 128
    x = 128
    # channels 6 and 7 are empty
    model1 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_2/inceptionunet_checkpoint_target_ch2.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model2 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_3/inceptionunet_checkpoint_target_ch3.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model3 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_4/inceptionunet_checkpoint_target_ch4.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model4 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_5/inceptionunet_checkpoint_target_ch5.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model5 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_8/inceptionunet_checkpoint_target_ch8.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model6 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_10/inceptionunet_checkpoint_target_ch10.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")
    model7 = createInceptionUnet(pretrained_weights=BASEPATH + "channel_11/inceptionunet_checkpoint_target_ch11.h5",
    input_shape=(y,x,3), n_labels=1,
    numFilters=numFilters, output_mode="sigmoid")

    # compile models for different target channels
    model1.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model2.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model3.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model4.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model5.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model6.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    model7.compile(loss=loss, optimizer="Adam", metrics=["accuracy", "mse"])
    print("Compiled model")

    #########################################################
    ################### EXTRACT FEATURES ####################
    #########################################################
    
    with h5py.File(IMAGE_PATH, "r") as file:
        file_length = len(file["channel_1/images"])
    
    # Go through each channel
    for current_channel in target_channels:
        if current_channel == 2:
            model = model1
        if current_channel == 3:
            model = model2
        if current_channel == 4:
            model = model3
        if current_channel == 5:
            model = model4
        if current_channel == 8:
            model = model5
        if current_channel == 10:
            model = model6
        if current_channel == 11:
            model = model7

        if not os.path.exists(SAVEPATH + "channel_{}/features".format(str(current_channel))):
            os.makedirs(SAVEPATH + "channel_{}/features".format(str(current_channel)))

        # go through each layer
        for layer_number, layer in enumerate(model.layers):
            wanted_layer_shape = (None, 1, 1, 16*numFilters)

            if layer_number >= (wanted_layer + 2):
                break

            if "conv" not in layer.name:
                continue # loop starts over, next layer

            if layer.output_shape != wanted_layer_shape: # notice that wanted layer number is 155 and shape is (None, 1, 1, 512)
                continue

            featuremodel = Model(inputs=model.inputs, outputs=model.layers[layer_number].output) # create new model for bottleneck of InceptionUnet
            generator = umap_generator(IMAGE_PATH, batch_size=batch_size) 
            batch_number = 0 
            with open(SAVEPATH + "channel_{}/features/featuremaps_layer_{}_reshaped_{}.csv".format(str(current_channel),str(layer_number),str(dataset)), "w+") as outfile:
                for batch in generator: # (5, 128, 128, 3)
                    if batch_number >= ((file_length / batch_size)): # was +1
                            break

                    feature_maps = featuremodel.predict(batch)
                    reshaped_features = feature_maps.reshape(len(batch), 1*1*(16*numFilters))
                    np.savetxt(outfile, reshaped_features, delimiter="\t")
                    batch_number += 1

    ############################################# PROCESSING FEATURE FILES #################################################################

    path_list = []
    for i, ch in enumerate(target_channels):
        df_path = SAVEPATH + "channel_{}/features/featuremaps_layer_{}_reshaped_{}.csv".format(str(ch),str(wanted_layer),str(dataset))
        path_list.append(df_path)

    df_list = []
    for path in path_list:
        df = pd.read_csv(path, sep="\t", header=None, dtype=float)
        df_list.append(df)

    # fill NaNs with zeros, remove all null rows and standardize all dfs
    for df in df_list:
        df = (df - df.mean()) / df.std()
        df = df[(df.T != 0).any()]
        df = df.fillna("")
        df = df.astype("float64")

    df_combined = pd.concat(df_list, ignore_index=True, axis=1)
    df_combined.to_csv(SAVEPATH + "layer{}_image_df_combined_{}.csv".format(str(wanted_layer), str(dataset)), sep="\t", header=False, index=False) 

    ######################################### IMPORTING IMAGES FOR INTERACTIVE UMAP #############################################################

    umap_images = h5py.File(IMAGE_PATH, "r")
    channels = list(umap_images.keys())
    channels = naturalsorting(channels)
    shape = tuple([len(channels)] + list(umap_images["channel_1/images"][0:data_point_amount].shape)) 
    image_matrix = np.empty(shape=shape, dtype=np.float32)

    for i, chan in enumerate(channels):
        ims = umap_images[chan]["images"][0:data_point_amount]
        image_matrix[i,:,:,:] = ims

    ch_x = image_matrix[(0,8,11),:data_point_amount,:,:] # channels 1, 9, 12 always
    ch_y = image_matrix[(list(map(lambda x: x - 1, target_channels))),:data_point_amount,:,:] #reduce 1 from each channel due to zero-based indexing

    ####################### Normalizing for each image ##########################

    norm_ch_x = np.zeros((ch_x.shape)) 
    norm_ch_y = np.zeros((ch_y.shape)) 

    for number, channel in enumerate(ch_x):
        for i, image in enumerate(channel):
            image = normalize(image) 
            norm_ch_x[number,i,:,:] = image

    for number, channel in enumerate(ch_y):
        for i, image in enumerate(channel):
            image = normalize(image)
            if norm_background == True and np.max(image)>0: 
                image = normalize_background(image, quantile)
            norm_ch_y[number,i,:,:] = image

    test_X = np.moveaxis(norm_ch_x, 0, 3)
    test_y = np.moveaxis(norm_ch_y, 0, 3)

    def embeddable_image(data):
        image = Image.fromarray((data*255).astype("uint8"), mode="L")
        buffer = BytesIO()
        image.save(buffer, format="png")
        for_encoding = buffer.getvalue()
        return "data:image/png;base64," + base64.b64encode(for_encoding).decode()

    indexlist = np.array(range(0, len(test_X)))

    # IMAGES TO EMBED: INPUTS, TARGETS AND PREDICTION

    # inputs: reducing dimensions for displaying as image
    imgs_ch1 = test_X[:,:,:,0] 
    imgs_ch9 = test_X[:,:,:,1] 
    imgs_ch12 = test_X[:,:,:,2] 
    imgs_ch2 = test_y[:,:,:,0] # CD45
    imgs_ch3 = test_y[:,:,:,1] # CD3
    imgs_ch4 = test_y[:,:,:,2] # CD14
    imgs_ch5 = test_y[:,:,:,3] # 7AAD
    imgs_ch8 = test_y[:,:,:,4] # CD19
    imgs_ch10 = test_y[:,:,:,5] # CD8
    imgs_ch11 = test_y[:,:,:,6] # CD56
    preds_ch2 = model1.predict(test_X)[:,:,:,0]
    preds_ch3 = model2.predict(test_X)[:,:,:,0]
    preds_ch4 = model3.predict(test_X)[:,:,:,0]
    preds_ch5 = model4.predict(test_X)[:,:,:,0]
    preds_ch8 = model5.predict(test_X)[:,:,:,0]
    preds_ch10 = model6.predict(test_X)[:,:,:,0]
    preds_ch11 = model7.predict(test_X)[:,:,:,0]

    # create UMAP embedding
    reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
    embedding = reducer.fit_transform(df_combined)
    np.savetxt(SAVEPATH + "{}_{}_channel_embedding_layer_{}.txt".format(str(file_prefix), str(len(target_channels)), str(wanted_layer)), embedding, delimiter="\t")
    assert(np.all(embedding == reducer.embedding_))
    cropped_embedding = embedding[:len(indexlist),:]
    print("Embedding shape for static umap: ", embedding.shape) # x, y dimensions
    print("Embedding shape for interactive umap: ", cropped_embedding.shape)
    print("Saving UMAP in ", str(SAVEPATH))
    
    # initialize shorter list of factors for interactive umap
    pred_factors = cells_pred_strict["Predicted cell type"][:len(test_X)]
    target_factors = cells_target_strict["Target cell type"][:len(test_X)]
    pred_factors = pred_factors.values.tolist()
    target_factors = target_factors.values.tolist()

    for color_mapping in color_mapping_list:
        ############## CREATE INTERACTIVE UMAP FOR TARGETS ############
        sns.set(style="white", context="notebook", rc={"figure.figsize":(10,10)})
        output_notebook()
        output_file(SAVEPATH + "{}_interactive_umap_{}_channels_layer{}_targets_colormap_{}.html".format(str(file_prefix), str(len(target_channels)), str(wanted_layer),str(color_mapping)))

        # create umap dataframe
        umap_df = pd.DataFrame(cropped_embedding, columns=("x", "y"))
        umap_df["index"] = [str(x) for x in indexlist] 
        umap_df["brightfield1"] = list(map(embeddable_image, imgs_ch1)) 
        umap_df["brightfield2"] = list(map(embeddable_image, imgs_ch9)) 
        umap_df["darkfield"] = list(map(embeddable_image, imgs_ch12)) 
        umap_df["channel2"] = list(map(embeddable_image, imgs_ch2)) 
        umap_df["channel3"] = list(map(embeddable_image, imgs_ch3))
        umap_df["channel4"] = list(map(embeddable_image, imgs_ch4))
        umap_df["channel5"] = list(map(embeddable_image, imgs_ch5))
        umap_df["channel8"] = list(map(embeddable_image, imgs_ch8))
        umap_df["channel10"] = list(map(embeddable_image, imgs_ch10)) 
        umap_df["channel11"] = list(map(embeddable_image, imgs_ch11)) 
        umap_df["preds_ch2"] = list(map(embeddable_image, preds_ch2))
        umap_df["preds_ch3"] = list(map(embeddable_image, preds_ch3))
        umap_df["preds_ch4"] = list(map(embeddable_image, preds_ch4))
        umap_df["preds_ch5"] = list(map(embeddable_image, preds_ch5))
        umap_df["preds_ch8"] = list(map(embeddable_image, preds_ch8))
        umap_df["preds_ch10"] = list(map(embeddable_image, preds_ch10)) 
        umap_df["preds_ch11"] = list(map(embeddable_image, preds_ch11))
        umap_df["CD45_means"] = list([float(x) for x in CD45_target[:len(test_X)]])
        umap_df["CD3_means"] = list([float(x) for x in CD3_target[:len(test_X)]])
        umap_df["CD14_means"] = list([float(x) for x in CD14_target[:len(test_X)]])
        umap_df["means_7AAD"] = list([float(x) for x in target_7AAD[:len(test_X)]])
        umap_df["CD19_means"] = list([float(x) for x in CD19_target[:len(test_X)]])
        umap_df["CD8_means"] = list([float(x) for x in CD8_target[:len(test_X)]])
        umap_df["CD56_means"] = list([float(x) for x in CD56_target[:len(test_X)]])
        umap_df["pred_factors"] = list([str(x) for x in pred_factors])
        umap_df["CD45_means_pred"] = list([float(x) for x in CD45_prediction[:len(test_X)]])
        umap_df["CD3_means_pred"] = list([float(x) for x in CD3_prediction[:len(test_X)]])
        umap_df["CD14_means_pred"] = list([float(x) for x in CD14_prediction[:len(test_X)]])
        umap_df["means_7AAD_pred"] = list([float(x) for x in prediction_7AAD[:len(test_X)]])
        umap_df["CD19_means_pred"] = list([float(x) for x in CD19_prediction[:len(test_X)]])
        umap_df["CD8_means_pred"] = list([float(x) for x in CD8_prediction[:len(test_X)]])
        umap_df["CD56_means_pred"] = list([float(x) for x in CD56_prediction[:len(test_X)]])
        umap_df["pred_factors"] = list([str(x) for x in pred_factors])
        umap_df["target_factors"] = list([str(x) for x in target_factors])
        datasource = ColumnDataSource(dict(umap_df))

        umap_plot = figure(
            title="UMAP projection of the dataset, target channels {}, color mapping for {}".format(str(target_channels), str(color_mapping)),
            plot_width=800,
            plot_height=800,
            tools=("pan, wheel_zoom, reset")
        )

        umap_plot.add_tools(HoverTool(tooltips="""
        <div>
            <div style='display:flex; flex-direction:column;'>
                <div style='display:flex; flex-direction:row;'>
                    <img src='@brightfield1' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@brightfield2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@darkfield' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                </div>
                <div style='display:flex; flex-direction:row;'>
                    <img src='@channel2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel3' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel4' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel5' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel8' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel10' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel11' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                </div>
                <div style='display:flex; flex-direction:row;'>
                    <img src='@preds_ch2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch3' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch4' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch5' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch8' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch10' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch11' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                </div>
            </div>
            <div>
                <span style='font-size: 12px; color: #224499'>Index:</span>
                <span style='font-size: 12px'>@index</span>
                <span style='font-size: 12px; color: #224499'>CD45 mean value:</span>
                <span style='font-size: 12px'>@CD45_means</span>
                <span style='font-size: 12px; color: #224499'>CD3 mean value:</span>
                <span style='font-size: 12px'>@CD3_means</span>
                <span style='font-size: 12px; color: #224499'>CD14 mean value:</span>
                <span style='font-size: 12px'>@CD14_means</span>
                <span style='font-size: 12px; color: #224499'>7AAD mean value:</span>
                <span style='font-size: 12px'>@means_7AAD</span>
                <span style='font-size: 12px; color: #224499'>CD19 mean value:</span>
                <span style='font-size: 12px'>@CD19_means</span>
                <span style='font-size: 12px; color: #224499'>CD8 mean value:</span>
                <span style='font-size: 12px'>@CD8_means</span>
                <span style='font-size: 12px; color: #224499'>CD56 mean value:</span>
                <span style='font-size: 12px'>@CD56_means</span>
                <span style='font-size: 12px; color: #224499'>CD45 pred value:</span>
                <span style='font-size: 12px'>@CD45_means_pred</span>
                <span style='font-size: 12px; color: #224499'>CD3 pred value:</span>
                <span style='font-size: 12px'>@CD3_means_pred</span>
                <span style='font-size: 12px; color: #224499'>CD14 pred value:</span>
                <span style='font-size: 12px'>@CD14_means_pred</span>
                <span style='font-size: 12px; color: #224499'>7AAD pred value:</span>
                <span style='font-size: 12px'>@means_7AAD_pred</span>
                <span style='font-size: 12px; color: #224499'>CD19 pred value:</span>
                <span style='font-size: 12px'>@CD19_means_pred</span>
                <span style='font-size: 12px; color: #224499'>CD8 pred value:</span>
                <span style='font-size: 12px'>@CD8_means_pred</span>
                <span style='font-size: 12px; color: #224499'>CD56 pred value:</span>
                <span style='font-size: 12px'>@CD56_means_pred</span>
                <span style='font-size: 12px; color: #224499'>Cell type predicted:</span>
                <span style='font-size: 12px'>@pred_factors</span>
                <span style='font-size: 12px; color: #224499'>Cell type target:</span>
                <span style='font-size: 12px'>@target_factors</span>
            </div>
            <div>
                <span style='font-size: 12px; color: #224499'>Left to right:</span>
                <span style='font-size: 12px'>bf1 ch1, bf2 ch9, df ch12, ch2-ch11 targets, ch2-11 pred</span>
            </div>

        </div>
        """))

        palette = Viridis[11]

        if color_mapping == "CD3": 
            color_mapper = linear_cmap(field_name="CD3_means", palette=palette, low=np.quantile(CD3_target,.1), high=np.quantile(CD3_target,.9))
        elif color_mapping == "CD45":
            color_mapper = linear_cmap(field_name="CD45_means", palette=palette, low=np.quantile(CD45_target,.1), high=np.quantile(CD45_target,.9))
        elif color_mapping == "CD14":
            color_mapper = linear_cmap(field_name="CD14_means", palette=palette, low=np.quantile(CD14_target,.1), high=np.quantile(CD14_target,.9))
        elif color_mapping == "7AAD":
            color_mapper = linear_cmap(field_name="means_7AAD", palette=palette, low=np.quantile(target_7AAD,.1), high=np.quantile(target_7AAD,.9))
        elif color_mapping == "CD19":
            color_mapper = linear_cmap(field_name="CD19_means", palette=palette, low=np.quantile(CD19_target,.1), high=np.quantile(CD19_target,.9))
        elif color_mapping == "CD8":
            color_mapper = linear_cmap(field_name="CD8_means", palette=palette, low=np.quantile(CD8_target,.1), high=np.quantile(CD8_target,.9))
        elif color_mapping == "CD56":
            color_mapper = linear_cmap(field_name="CD56_means", palette=palette, low=np.quantile(CD56_target,.1), high=np.quantile(CD56_target,.9))
        else: 
            color_mapper = linear_cmap(field_name="CD45_means", palette=palette, low=np.quantile(CD45_target,.1), high=np.quantile(CD45_target,.9))
        umap_plot.circle(
        "x",
        "y",
        source=datasource,
        line_color=color_mapper,
        fill_color=color_mapper,
        line_alpha=0.8,
        fill_alpha=0.8,
        size=6)
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        show(umap_plot)

        ############## CREATE INTERACTIVE UMAP FOR PREDICTIONS ############
        sns.set(style="white", context="notebook", rc={"figure.figsize":(10,10)})
        output_notebook()
        output_file(SAVEPATH + "{}_interactive_umap_{}_channels_layer{}_predictions_colormap_{}.html".format(str(file_prefix), str(len(target_channels)), str(wanted_layer),str(color_mapping)))

        # create umap dataframe
        umap_df = pd.DataFrame(cropped_embedding, columns=("x", "y"))
        umap_df["index"] = [str(x) for x in indexlist] 
        umap_df["brightfield1"] = list(map(embeddable_image, imgs_ch1))
        umap_df["brightfield2"] = list(map(embeddable_image, imgs_ch9))
        umap_df["darkfield"] = list(map(embeddable_image, imgs_ch12))
        umap_df["channel2"] = list(map(embeddable_image, imgs_ch2))
        umap_df["channel3"] = list(map(embeddable_image, imgs_ch3))
        umap_df["channel4"] = list(map(embeddable_image, imgs_ch4))
        umap_df["channel5"] = list(map(embeddable_image, imgs_ch5))
        umap_df["channel8"] = list(map(embeddable_image, imgs_ch8))
        umap_df["channel10"] = list(map(embeddable_image, imgs_ch10))
        umap_df["channel11"] = list(map(embeddable_image, imgs_ch11))
        umap_df["CD45_means"] = list([float(x) for x in CD45_target[:len(test_X)]])
        umap_df["CD3_means"] = list([float(x) for x in CD3_target[:len(test_X)]])
        umap_df["CD14_means"] = list([float(x) for x in CD14_target[:len(test_X)]])
        umap_df["means_7AAD"] = list([float(x) for x in target_7AAD[:len(test_X)]])
        umap_df["CD19_means"] = list([float(x) for x in CD19_target[:len(test_X)]])
        umap_df["CD8_means"] = list([float(x) for x in CD8_target[:len(test_X)]])
        umap_df["CD56_means"] = list([float(x) for x in CD56_target[:len(test_X)]])
        umap_df["preds_ch2"] = list(map(embeddable_image, preds_ch2))
        umap_df["preds_ch3"] = list(map(embeddable_image, preds_ch3))
        umap_df["preds_ch4"] = list(map(embeddable_image, preds_ch4))
        umap_df["preds_ch5"] = list(map(embeddable_image, preds_ch5))
        umap_df["preds_ch8"] = list(map(embeddable_image, preds_ch8))
        umap_df["preds_ch10"] = list(map(embeddable_image, preds_ch10))
        umap_df["preds_ch11"] = list(map(embeddable_image, preds_ch11))
        umap_df["CD45_means_pred"] = list([float(x) for x in CD45_prediction[:len(test_X)]])
        umap_df["CD3_means_pred"] = list([float(x) for x in CD3_prediction[:len(test_X)]])
        umap_df["CD14_means_pred"] = list([float(x) for x in CD14_prediction[:len(test_X)]])
        umap_df["means_7AAD_pred"] = list([float(x) for x in prediction_7AAD[:len(test_X)]])
        umap_df["CD19_means_pred"] = list([float(x) for x in CD19_prediction[:len(test_X)]])
        umap_df["CD8_means_pred"] = list([float(x) for x in CD8_prediction[:len(test_X)]])
        umap_df["CD56_means_pred"] = list([float(x) for x in CD56_prediction[:len(test_X)]])
        umap_df["pred_factors"] = list([str(x) for x in pred_factors])
        umap_df["target_factors"] = list([str(x) for x in target_factors])
        datasource = ColumnDataSource(dict(umap_df))

        umap_plot = figure(
            title="UMAP projection of the dataset, target channels {}, color mapping for {}, prediction".format(str(target_channels), str(color_mapping)),
            plot_width=800,
            plot_height=800,
            tools=("pan, wheel_zoom, reset")
        )

        umap_plot.add_tools(HoverTool(tooltips="""
        <div>
            <div style='display:flex; flex-direction:column;'>
                <div style='display:flex; flex-direction:row;'>
                    <img src='@brightfield1' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@brightfield2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@darkfield' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                </div>
                <div style='display:flex; flex-direction:row;'>
                    <img src='@channel2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel3' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel4' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel5' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel8' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel10' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@channel11' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                </div>
                <div style='display:flex; flex-direction:row;'>
                    <img src='@preds_ch2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch3' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch4' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch5' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch8' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch10' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                    <img src='@preds_ch11' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
                </div>
            </div>
            <div>
                <span style='font-size: 12px; color: #224499'>Index:</span>
                <span style='font-size: 12px'>@index</span>
                <span style='font-size: 12px; color: #224499'>CD45 mean value:</span>
                <span style='font-size: 12px'>@CD45_means</span>
                <span style='font-size: 12px; color: #224499'>CD3 mean value:</span>
                <span style='font-size: 12px'>@CD3_means</span>
                <span style='font-size: 12px; color: #224499'>CD14 mean value:</span>
                <span style='font-size: 12px'>@CD14_means</span>
                <span style='font-size: 12px; color: #224499'>7AAD mean value:</span>
                <span style='font-size: 12px'>@means_7AAD</span>
                <span style='font-size: 12px; color: #224499'>CD19 mean value:</span>
                <span style='font-size: 12px'>@CD19_means</span>
                <span style='font-size: 12px; color: #224499'>CD8 mean value:</span>
                <span style='font-size: 12px'>@CD8_means</span>
                <span style='font-size: 12px; color: #224499'>CD56 mean value:</span>
                <span style='font-size: 12px'>@CD56_means</span>
                <span style='font-size: 12px; color: #224499'>CD45 pred value:</span>
                <span style='font-size: 12px'>@CD45_means_pred</span>
                <span style='font-size: 12px; color: #224499'>CD3 pred value:</span>
                <span style='font-size: 12px'>@CD3_means_pred</span>
                <span style='font-size: 12px; color: #224499'>CD14 pred value:</span>
                <span style='font-size: 12px'>@CD14_means_pred</span>
                <span style='font-size: 12px; color: #224499'>7AAD pred value:</span>
                <span style='font-size: 12px'>@means_7AAD_pred</span>
                <span style='font-size: 12px; color: #224499'>CD19 pred value:</span>
                <span style='font-size: 12px'>@CD19_means_pred</span>
                <span style='font-size: 12px; color: #224499'>CD8 pred value:</span>
                <span style='font-size: 12px'>@CD8_means_pred</span>
                <span style='font-size: 12px; color: #224499'>CD56 pred value:</span>
                <span style='font-size: 12px'>@CD56_means_pred</span>
                <span style='font-size: 12px; color: #224499'>Cell type predicted:</span>
                <span style='font-size: 12px'>@pred_factors</span>
                <span style='font-size: 12px; color: #224499'>Cell type target:</span>
                <span style='font-size: 12px'>@target_factors</span>
            </div>
            <div>
                <span style='font-size: 12px; color: #224499'>Left to right:</span>
                <span style='font-size: 12px'>bf1 ch1, bf2 ch9, df ch12, ch2-ch11 targets, ch2-11 pred</span>
            </div>

        </div>
        """)) 

        palette = Viridis[11]
        if color_mapping == "CD3": 
            color_mapper = linear_cmap(field_name="CD3_means_pred", palette=palette, low=np.quantile(CD3_prediction,.1), high=np.quantile(CD3_prediction,.9))
        elif color_mapping == "CD45":
            color_mapper = linear_cmap(field_name="CD45_means_pred", palette=palette, low=np.quantile(CD45_prediction,.1), high=np.quantile(CD45_prediction,.9))
        elif color_mapping == "CD14":
            color_mapper = linear_cmap(field_name="CD14_means_pred", palette=palette, low=np.quantile(CD14_prediction,.1), high=np.quantile(CD14_prediction,.9))
        elif color_mapping == "7AAD":
            color_mapper = linear_cmap(field_name="means_7AAD_pred", palette=palette, low=np.quantile(prediction_7AAD,.1), high=np.quantile(prediction_7AAD,.9))
        elif color_mapping == "CD19":
            color_mapper = linear_cmap(field_name="CD19_means_pred", palette=palette, low=np.quantile(CD19_prediction,.1), high=np.quantile(CD19_prediction,.9))
        elif color_mapping == "CD8":
            color_mapper = linear_cmap(field_name="CD8_means_pred", palette=palette, low=np.quantile(CD8_prediction,.1), high=np.quantile(CD8_prediction,.9))
        elif color_mapping == "CD56":
            color_mapper = linear_cmap(field_name="CD56_means_pred", palette=palette, low=np.quantile(CD56_prediction,.1), high=np.quantile(CD56_prediction,.9))
        else: 
            color_mapper = linear_cmap(field_name="CD45_means_pred", palette=palette, low=np.quantile(CD45_prediction,.1), high=np.quantile(CD45_prediction,.9))
        umap_plot.circle(
        "x",
        "y",
        source=datasource,
        line_color=color_mapper,
        fill_color=color_mapper,
        line_alpha=0.8,
        fill_alpha=0.8,
        size=6)
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        show(umap_plot)

# create static umaps
embedding = embedding[:file_length]
colormap_pred = plt.cm.get_cmap("viridis")
colormap_target = plt.cm.get_cmap("viridis")
alpha = 0.3
size = 2

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD45_target,low_quantile), vmax=np.quantile(CD45_target,high_quantile))

# CD45
plt.figure() 
plt.scatter(embedding[:,0], embedding[:,1], c=CD45_target, norm=normalize, cmap=colormap_pred, s=size, alpha=alpha) 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD45 target".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD45_target.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD3_target,low_quantile), vmax=np.quantile(CD3_target,high_quantile))

# CD3
plt.figure()
plt.scatter(embedding[:,0], embedding[:,1], c=CD3_target, norm=normalize, cmap=colormap_pred, s=size, alpha=alpha)
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD3 target".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD3_target.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD14_target,low_quantile), vmax=np.quantile(CD14_target,high_quantile))
# CD14 
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD14_target, norm=normalize, cmap=colormap_pred, s=size, alpha=alpha)
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD14 target".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.tight_layout(h_pad=1)
plt.colorbar(label="Intensity from low to high")
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD14_target.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(target_7AAD,low_quantile), vmax=np.quantile(target_7AAD,high_quantile))
# 7AAD
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=target_7AAD, norm=normalize, cmap=colormap_pred, s=size, alpha=alpha)
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, 7AAD target".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_7AAD_target.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD19_target,low_quantile), vmax=np.quantile(CD19_target,high_quantile))
# CD19
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD19_target, norm=normalize, cmap=colormap_pred, s=size, alpha=alpha) #.reversed()
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD19 target".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD19_target.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD8_target,low_quantile), vmax=np.quantile(CD8_target,high_quantile))
# CD8
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD8_target, norm=normalize, cmap=colormap_pred, s=size, alpha=alpha)
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD8 target".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD8_target.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD56_target,low_quantile), vmax=np.quantile(CD56_target,high_quantile))
# CD56
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD56_target, norm=normalize, cmap=colormap_pred, s=size, alpha=alpha) 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD56 target".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD56_target.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

############################################################
###### STATIC UMAP FOR PREDICTIONS #########################
############################################################
normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD45_prediction,low_quantile), vmax=np.quantile(CD45_prediction,high_quantile)) #np.quantile(CD45_target,low_quantile)
# CD45 
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD45_prediction, norm=normalize, cmap=colormap_target, s=size, alpha=alpha) #norm=normalize, 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD45 prediction".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD45_pred.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD3_prediction,low_quantile), vmax=np.quantile(CD3_prediction,high_quantile))
# CD3
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD3_prediction, norm=normalize, cmap=colormap_target, s=size, alpha=alpha) 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD3 prediction".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD3_pred.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD14_prediction,low_quantile), vmax=np.quantile(CD14_prediction,high_quantile))
# CD14 
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD14_prediction, norm=normalize, cmap=colormap_target, s=size, alpha=alpha) 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD14 prediction".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD14_pred.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(prediction_7AAD,low_quantile), vmax=np.quantile(prediction_7AAD,high_quantile))
# 7AAD
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=prediction_7AAD, norm=normalize, cmap=colormap_target, s=size, alpha=alpha) 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, 7AAD prediction".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_7AAD_pred.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD19_prediction,low_quantile), vmax=np.quantile(CD19_prediction,high_quantile))
# CD19
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD19_prediction, norm=normalize, cmap=colormap_target, s=size, alpha=alpha) 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD19 prediction".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD19_pred.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD8_prediction,low_quantile), vmax=np.quantile(CD8_prediction,high_quantile))
# CD8
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD8_prediction, norm=normalize, cmap=colormap_target, s=size, alpha=alpha) 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD8 prediction".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD8_pred.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

normalize = matplotlib.colors.Normalize(vmin=np.quantile(CD56_prediction,low_quantile), vmax=np.quantile(CD56_prediction,high_quantile))
# CD56
plt.figure() # figsize=(10,10)
plt.scatter(embedding[:,0], embedding[:,1], c=CD56_prediction, norm=normalize, cmap=colormap_target, s=size, alpha=alpha) 
plt.title("UMAP projection of the {} dataset, {} channels, n_neighbors={}, CD56 prediction".format(str(dataset), str(len(target_channels)), str(n_neighbors)), fontsize=8)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.colorbar(label="Intensity from low to high")
plt.tight_layout(h_pad=1)
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_CD56_pred.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()


# create static umap for cell type color mapping
cell_type_factors = cells_target_strict["Target cell type"]
print(cell_type_factors.unique())
cell_type_factors = cell_type_factors.values.tolist()

numerical_cell_types = []
for cell in cell_type_factors:
    if cell == "Damaged/Dead":
        numerical_cell_types.append(0)
    elif cell == "T Cell":
        numerical_cell_types.append(1)
    elif cell == "Monocyte":
        numerical_cell_types.append(2)
    elif cell == "B Cell":
        numerical_cell_types.append(3)
    elif cell == "NK":
        numerical_cell_types.append(4)
    elif cell == "Cytotoxic T Cell":
        numerical_cell_types.append(5)
    elif cell == "NKT":
        numerical_cell_types.append(6)
    elif cell == "WBC":
        numerical_cell_types.append(7)
    elif cell == "Unknown":
        numerical_cell_types.append(8)


type_factor_palette = {"Damaged/Dead":"red","T Cell":"blue","Monocyte":"green","B Cell":"lime","NK":"brown","Cytotoxic T Cell":"yellow","NKT":"gray","WBC":"pink","Unknown":"purple"}
embedding_df = pd.DataFrame(dict(x=embedding[:,0], y=embedding[:,1], categories=cell_type_factors))
fig = plt.figure()
plt.scatter(embedding_df["x"], embedding_df["y"], c=embedding_df["categories"].map(type_factor_palette), s=size, alpha=alpha) 
plt.tight_layout(h_pad=1)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.savefig(SAVEPATH + "{}_{}_channel_umap_layer{}_celltypes.png".format(str(file_prefix),str(len(target_channels)),str(wanted_layer)))
plt.close()

# initialize shorter list of factors for interactive umap
cell_type_factors = cell_type_factors[:len(indexlist)]

############## CREATE INTERACTIVE UMAP FOR CELLTYPE COLOR MAPPING ############
sns.set(style="white", context="notebook", rc={"figure.figsize":(10,10)})
output_notebook()
output_file(SAVEPATH + "{}_interactive_umap_{}_channels_layer{}_colormap_celltype.html".format(str(file_prefix), str(len(target_channels)), str(wanted_layer)))

# create umap dataframe
umap_df = pd.DataFrame(cropped_embedding, columns=("x", "y"))
umap_df["index"] = [str(x) for x in indexlist] 
umap_df["brightfield1"] = list(map(embeddable_image, imgs_ch1))
umap_df["brightfield2"] = list(map(embeddable_image, imgs_ch9))
umap_df["darkfield"] = list(map(embeddable_image, imgs_ch12))
umap_df["channel2"] = list(map(embeddable_image, imgs_ch2))
umap_df["channel3"] = list(map(embeddable_image, imgs_ch3))
umap_df["channel4"] = list(map(embeddable_image, imgs_ch4))
umap_df["channel5"] = list(map(embeddable_image, imgs_ch5))
umap_df["channel8"] = list(map(embeddable_image, imgs_ch8))
umap_df["channel10"] = list(map(embeddable_image, imgs_ch10))
umap_df["channel11"] = list(map(embeddable_image, imgs_ch11))
umap_df["preds_ch2"] = list(map(embeddable_image, preds_ch2))
umap_df["preds_ch3"] = list(map(embeddable_image, preds_ch3))
umap_df["preds_ch4"] = list(map(embeddable_image, preds_ch4))
umap_df["preds_ch5"] = list(map(embeddable_image, preds_ch5))
umap_df["preds_ch8"] = list(map(embeddable_image, preds_ch8))
umap_df["preds_ch10"] = list(map(embeddable_image, preds_ch10))
umap_df["preds_ch11"] = list(map(embeddable_image, preds_ch11))
umap_df["CD45_means"] = list([float(x) for x in CD45_target[:len(test_X)]])
umap_df["CD3_means"] = list([float(x) for x in CD3_target[:len(test_X)]])
umap_df["CD14_means"] = list([float(x) for x in CD14_target[:len(test_X)]])
umap_df["means_7AAD"] = list([float(x) for x in target_7AAD[:len(test_X)]])
umap_df["CD19_means"] = list([float(x) for x in CD19_target[:len(test_X)]])
umap_df["CD8_means"] = list([float(x) for x in CD8_target[:len(test_X)]])
umap_df["CD56_means"] = list([float(x) for x in CD56_target[:len(test_X)]])
umap_df["cell_type_factors"] = list([str(x) for x in cell_type_factors])
datasource = ColumnDataSource(dict(umap_df))

umap_plot = figure(
    title="UMAP projection of the dataset, target channels {}, color mapping for cell type".format(str(target_channels)),
    plot_width=800,
    plot_height=800,
    tools=("pan, wheel_zoom, reset")
)

umap_plot.add_tools(HoverTool(tooltips="""
<div>
    <h1 style='font-size: 24px; color: #224499'>@cell_type_factors<h1>
    <div style='display:flex; flex-direction:column;'>
        <div style='display:flex; flex-direction:row;'>
            <img src='@brightfield1' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@brightfield2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@darkfield' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div style='display:flex; flex-direction:row;'>
            <img src='@channel2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@channel3' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@channel4' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@channel5' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@channel8' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@channel10' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@channel11' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div style='display:flex; flex-direction:row;'>
            <img src='@preds_ch2' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@preds_ch3' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@preds_ch4' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@preds_ch5' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@preds_ch8' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@preds_ch10' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
            <img src='@preds_ch11' style='max-width: 60px; max-height: 60px; float: left; margin: 5px 5px 5px 5px'/>
        </div>
    </div>
    <div>
        <span style='font-size: 12px; color: #224499'>Index:</span>
        <span style='font-size: 12px'>@index</span>
        <span style='font-size: 12px; color: #224499'>CD45 mean value:</span>
        <span style='font-size: 12px'>@CD45_means</span>
        <span style='font-size: 12px; color: #224499'>CD3 mean value:</span>
        <span style='font-size: 12px'>@CD3_means</span>
        <span style='font-size: 12px; color: #224499'>CD14 mean value:</span>
        <span style='font-size: 12px'>@CD14_means</span>
        <span style='font-size: 12px; color: #224499'>7AAD mean value:</span>
        <span style='font-size: 12px'>@means_7AAD</span>
        <span style='font-size: 12px; color: #224499'>CD19 mean value:</span>
        <span style='font-size: 12px'>@CD19_means</span>
        <span style='font-size: 12px; color: #224499'>CD8 mean value:</span>
        <span style='font-size: 12px'>@CD8_means</span>
        <span style='font-size: 12px; color: #224499'>CD56 mean value:</span>
        <span style='font-size: 12px'>@CD56_means</span>
        <span style='font-size: 12px; color: #224499'>Cell type:</span>
        <span style='font-size: 12px'>@cell_type_factors</span>
    </div>
    <div>
        <span style='font-size: 12px; color: #224499'>Left to right:</span>
        <span style='font-size: 12px'>bf1 ch1, bf2 ch9, df ch12, ch2-ch11 targets, ch2-11 pred</span>
    </div>

</div>
"""))

color_mapper = factor_cmap("cell_type_factors", palette=["red","blue","yellow","green","lime","brown","yellow","gray","pink","purple"], factors=umap_df["cell_type_factors"].unique())
umap_plot.circle(
"x",
"y",
source=datasource,
line_color=color_mapper,
fill_color=color_mapper,
line_alpha=0.8,
fill_alpha=0.8,
size=6)
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
show(umap_plot)