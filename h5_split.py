#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions to split, merge or fill HDF datasets to size.

ONLY takes in 12 channel h5 files. It is unable to work with fractions, int values must be used.

If you do not want either a validation or test set in merge function, leave the corresponding parameter as 0. Give as parameters the number of images you want as train, validation and test. 

Example for merging:

python h5_split.py --merge_file1 /path/to/file/TEST.h5 --merge_file2 /path/to/file/TEST2.h5 --save_path /path/to/folder/ --output_name testfile_ --wanted_function merge

Example for splitting:
python h5_split.py --hdf_file /path/to/file/TEST.h5 --save_path /path/to/folder/ --val 10000 --test 10000 --output_name testfile_ --wanted_function split

Example for filling to size:
python h5_split.py --hdf_file /path/to/file/TEST.h5 --save_path /path/to/folder/ --output_name testfile_ --wanted_function fill_to_size

"""
import argparse
import h5py
import numpy as np
import random
import re

random.seed(42)

parser = argparse.ArgumentParser(description="Functions for modifying HDF files.")
parser.add_argument("--hdf_file", type=str, help="If using split: path of hdf file to split, results go in split_h5_sets")
parser.add_argument("--save_path", type=str, help="Save path")
parser.add_argument("--output_name", type=str, help="Name of output")
parser.add_argument("--val", type=str, help="Validation image amount")
parser.add_argument("--test", type=str, help="Test image amount")
parser.add_argument("--merge_file1", type=str, help="If using merge, file 1 path")
parser.add_argument("--merge_file2", type=str, help="If using merge, file 2 path")
parser.add_argument("--wanted_function", type=str, help="Which function to use: split, merge or fill_to_size")
args = parser.parse_args()

hdf_file = args.hdf_file
save_path = args.save_path
output_name = args.output_name
test_amount = args.test
val_amount = args.val
file1_path = args.merge_file1
file2_path = args.merge_file2
wanted_function = args.wanted_function

def naturalsorting(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    number_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=number_key)

def split_train_test(file_path, save_path, test_amount, val_amount, output_name): 
    """
    Split a h5 file to parts, e.g. train, test and validation sets.
    """
    with h5py.File(file_path, "r") as file:
        image_amount = len(file["channel_1/images"])
        print("Data set length is ", image_amount)
        test_amount = int(test_amount)
        print("Test image amount is ", str(test_amount), " and percentage of full data is ", str(test_amount / image_amount))
        val_amount = int(val_amount)
        print("Validation image amount is ", str(val_amount), " and percentage of full data is ", str(val_amount / image_amount))
        train_amount = int(image_amount - test_amount - val_amount)
        channels = list(file.keys())

        # determining shapes of matrices for different datasets
        shape = tuple([len(channels)] + list(file["channel_1/images"][:(image_amount - (val_amount+test_amount))].shape))
        print("Shape of matrix 1: train set, ", shape)
        shape2 = tuple([len(channels)] + list(file["channel_1/images"][:test_amount].shape))
        print("Shape of matrix 2: test set, ", shape2)
        shape3 = tuple([len(channels)] + list(file["channel_1/images"][:val_amount].shape))
        print("Shape of matrix 3: val set, ", shape3)

        # creating empty matrices for images
        train_matrix = np.empty(shape=shape, dtype=np.float32) 
        test_matrix = np.empty(shape=shape2, dtype=np.float32) 
        val_matrix = np.empty(shape=shape3, dtype=np.float32) 

        for i, channel in enumerate(naturalsorting(channels)): 
            # indexing each channel until the end of training set 
            train_images = file["{}/images".format(str(channel))][:(image_amount - (val_amount+test_amount))]
            train_matrix[i,...] = train_images

            test_images = file["{}/images".format(str(channel))][(image_amount - (val_amount+test_amount)):(image_amount - (val_amount+test_amount) + test_amount)]
            test_matrix[i,...] = test_images

            val_images = file["{}/images".format(str(channel))][(image_amount - (val_amount+test_amount) + test_amount):(image_amount - (val_amount+test_amount) + test_amount + val_amount)]
            val_matrix[i,...] = val_images

        print("Train matrix shape ", train_matrix.shape)
        print("Test matrix shape ", test_matrix.shape)
        print("Val matrix shape ", val_matrix.shape)

    with h5py.File(save_path + "train_len{}_{}.h5".format(str(train_amount), str(output_name)), "w") as f: 
        ### TRAINING SET ###
        print("Processing training set")
        ch1 = f.create_group("channel_1")
        ch1.create_dataset("images", data=train_matrix[0,...])

        ch2 = f.create_group("channel_2")
        ch2.create_dataset("images", data=train_matrix[1,...])
        
        ch3 = f.create_group("channel_3")
        ch3.create_dataset("images", data=train_matrix[2,...])

        ch4 = f.create_group("channel_4")
        ch4.create_dataset("images", data=train_matrix[3,...])
        
        ch5 = f.create_group("channel_5")
        ch5.create_dataset("images", data=train_matrix[4,...])

        ch6 = f.create_group("channel_6")
        ch6.create_dataset("images", data=train_matrix[5,...])

        ch7 = f.create_group("channel_7")
        ch7.create_dataset("images", data=train_matrix[6,...])
        
        ch8 = f.create_group("channel_8")
        ch8.create_dataset("images", data=train_matrix[7,...])
        
        ch9 = f.create_group("channel_9")
        ch9.create_dataset("images", data=train_matrix[8,...])
        
        ch10 = f.create_group("channel_10")
        ch10.create_dataset("images", data=train_matrix[9,...])
        
        ch11 = f.create_group("channel_11")
        ch11.create_dataset("images", data=train_matrix[10,...])
        
        ch12 = f.create_group("channel_12")
        ch12.create_dataset("images", data=train_matrix[11,...])
        
    with h5py.File(save_path + "test_len{}_{}.h5".format(str(test_amount), str(output_name)), "w") as f: 
        ### TEST SET ####
        print("Processing test set")
        ch1 = f.create_group("channel_1")
        ch1.create_dataset("images", data=test_matrix[0,...])

        ch2 = f.create_group("channel_2")
        ch2.create_dataset("images", data=test_matrix[1,...])
        
        ch3 = f.create_group("channel_3")
        ch3.create_dataset("images", data=test_matrix[2,...])

        ch4 = f.create_group("channel_4")
        ch4.create_dataset("images", data=test_matrix[3,...])
        
        ch5 = f.create_group("channel_5")
        ch5.create_dataset("images", data=test_matrix[4,...])

        ch6 = f.create_group("channel_6")
        ch6.create_dataset("images", data=test_matrix[5,...])

        ch7 = f.create_group("channel_7")
        ch7.create_dataset("images", data=test_matrix[6,...])
        
        ch8 = f.create_group("channel_8")
        ch8.create_dataset("images", data=test_matrix[7,...])
        
        ch9 = f.create_group("channel_9") 
        ch9.create_dataset("images", data=test_matrix[8,...])
        
        ch10 = f.create_group("channel_10")
        ch10.create_dataset("images", data=test_matrix[9,...])

        ch11 = f.create_group("channel_11")
        ch11.create_dataset("images", data=test_matrix[10,...])
        
        ch12 = f.create_group("channel_12") 
        ch12.create_dataset("images", data=test_matrix[11,...])
        
    with h5py.File(save_path + "val_len{}_{}.h5".format(str(val_amount), str(output_name)), "w") as f: 
        ### VALIDATION SET ###
        print("Processing validation set")
        ch1 = f.create_group("channel_1")
        ch1.create_dataset("images", data=val_matrix[0,...])

        ch2 = f.create_group("channel_2")
        ch2.create_dataset("images", data=val_matrix[1,...])
        
        ch3 = f.create_group("channel_3")
        ch3.create_dataset("images", data=val_matrix[2,...])

        ch4 = f.create_group("channel_4")
        ch4.create_dataset("images", data=val_matrix[3,...])
        
        ch5 = f.create_group("channel_5")
        ch5.create_dataset("images", data=val_matrix[4,...])

        ch6 = f.create_group("channel_6")
        ch6.create_dataset("images", data=val_matrix[5,...])

        ch7 = f.create_group("channel_7")
        ch7.create_dataset("images", data=val_matrix[6,...])
        
        ch8 = f.create_group("channel_8")
        ch8.create_dataset("images", data=val_matrix[7,...])
        
        ch9 = f.create_group("channel_9")
        ch9.create_dataset("images", data=val_matrix[8,...])
        
        ch10 = f.create_group("channel_10")
        ch10.create_dataset("images", data=val_matrix[9,...])
        
        ch11 = f.create_group("channel_11")
        ch11.create_dataset("images", data=val_matrix[10,...])
        
        ch12 = f.create_group("channel_12") 
        ch12.create_dataset("images", data=val_matrix[11,...])


def merge_datasets(file1_path, file2_path, save_path, output_name):
    """
    Merge two HDF files into one dataset.
    """
    print("Opening data files")
    hf1 = h5py.File(file1_path)
    print("Opened file 1 ", file1_path)
    hf2 = h5py.File(file2_path)
    print("Opened file 2 ", file2_path)

    # Process first file and create new dataset to append to
    channels = list(hf1.keys())
    print("File 1 channels: ", channels)
    print("File 2 channels: ", list(hf2.keys()))
    shape1 = tuple([len(channels)] + list(hf1["channel_1/images"].shape))
    print(shape1) # (channels, images, height, width)
    hf1_matrix = np.empty(shape=shape1, dtype=np.float32) 

    for i, channel in enumerate(naturalsorting(channels)): 
        # indexing for each channel until the end of training set 
        images = hf1["{}/images".format(str(channel))]
        hf1_matrix[i,...] = images
    
    with h5py.File(save_path + output_name + ".h5", "a") as f:
        ch1 = f.create_group("channel_1")
        ch1.create_dataset("images", data=hf1_matrix[0,...], maxshape=(None, 128, 128))

        ch2 = f.create_group("channel_2")
        ch2.create_dataset("images", data=hf1_matrix[1,...], maxshape=(None, 128, 128))

        ch3 = f.create_group("channel_3")
        ch3.create_dataset("images", data=hf1_matrix[2,...], maxshape=(None, 128, 128))

        ch4 = f.create_group("channel_4")
        ch4.create_dataset("images", data=hf1_matrix[3,...], maxshape=(None, 128, 128))
        
        ch5 = f.create_group("channel_5")
        ch5.create_dataset("images", data=hf1_matrix[4,...], maxshape=(None, 128, 128))

        ch6 = f.create_group("channel_6")
        ch6.create_dataset("images", data=hf1_matrix[5,...], maxshape=(None, 128, 128))

        ch7 = f.create_group("channel_7")
        ch7.create_dataset("images", data=hf1_matrix[6,...], maxshape=(None, 128, 128))

        ch8 = f.create_group("channel_8")
        ch8.create_dataset("images", data=hf1_matrix[7,...], maxshape=(None, 128, 128))

        ch9 = f.create_group("channel_9")
        ch9.create_dataset("images", data=hf1_matrix[8,...], maxshape=(None, 128, 128))

        ch10 = f.create_group("channel_10")
        ch10.create_dataset("images", data=hf1_matrix[9,...], maxshape=(None, 128, 128))

        ch11 = f.create_group("channel_11")
        ch11.create_dataset("images", data=hf1_matrix[10,...], maxshape=(None, 128, 128))

        ch12 = f.create_group("channel_12")
        ch12.create_dataset("images", data=hf1_matrix[11,...], maxshape=(None, 128, 128))
        
    
    # Process the other file and append to existing dataset
    with h5py.File(save_path + output_name + ".h5", "a") as combined_file:
        print(combined_file)
        for channel in channels:
            print("Handling channel ", str(channel))
            data_to_add = hf2[channel]["images"]
            len_oldfile = combined_file[channel]["images"].shape[0] # (images, height, width)
            len_newfile = data_to_add.shape[0]
            new_len = int(len_oldfile) + int(len_newfile)
            new_shape = (new_len, 128, 128)
            combined_file[channel]["images"].resize((new_shape))
            print("Added data shape ", data_to_add.shape)
            print("New data shape ", combined_file[channel]["images"])
            combined_file[channel]["images"][-len_newfile:,...] = data_to_add
            print("New shape of file ", combined_file[channel]["images"].shape)

def fill_to_size(file_path, save_path, output_name):
    """
    Function to fill HDF channels to a new size using blank images, for compatibility with other datasets.
    Specific to RBC dataset from Doan et al. (2020). Change to fit the dataset. 
    RBC dataset has 4 channels while MNC data has 12; channels 1, 2, 4 are bf1, bf2 and df in the RBC data.
    """
    with h5py.File(file_path, "a") as hf:
        print("Opened file", file_path)
        # Process first file and create new dataset to append to
        channels = list(hf.keys())
        print("File 1 channels: ", channels)
        shape = tuple([len(channels)] + list(hf["channel_1/images"].shape))
        print(shape)
        wanted_size=(12,shape[1],128,128)
        print(wanted_size)
        ch1 = hf["channel_1"]["images"]
        ch9 = hf["channel_3"]["images"]
        ch2 = hf["channel_2"]["images"]
        ch12 = hf["channel_4"]["images"]
        blank_channel = np.zeros((shape[1],128,128))
        hf1_matrix = np.empty(shape=wanted_size, dtype=np.float32) 
        hf1_matrix[0,...] = ch1
        hf1_matrix[1,...] = ch2
        hf1_matrix[2,...] = blank_channel
        hf1_matrix[3,...] = blank_channel
        hf1_matrix[4,...] = blank_channel
        hf1_matrix[5,...] = blank_channel
        hf1_matrix[6,...] = blank_channel
        hf1_matrix[7,...] = blank_channel
        hf1_matrix[8,...] = ch9
        hf1_matrix[9,...] = blank_channel
        hf1_matrix[10,...] = blank_channel
        hf1_matrix[11,...] = ch12

        with h5py.File(save_path + output_name + ".h5", "a") as f:
            ch1 = f.create_group("channel_1")
            ch1.create_dataset("images", data=hf1_matrix[0,...])

            ch2 = f.create_group("channel_2")
            ch2.create_dataset("images", data=hf1_matrix[1,...])

            ch3 = f.create_group("channel_3")
            ch3.create_dataset("images", data=hf1_matrix[2,...])

            ch4 = f.create_group("channel_4")
            ch4.create_dataset("images", data=hf1_matrix[3,...])
            
            ch5 = f.create_group("channel_5")
            ch5.create_dataset("images", data=hf1_matrix[4,...])

            ch6 = f.create_group("channel_6")
            ch6.create_dataset("images", data=hf1_matrix[5,...])

            ch7 = f.create_group("channel_7")
            ch7.create_dataset("images", data=hf1_matrix[6,...])

            ch8 = f.create_group("channel_8")
            ch8.create_dataset("images", data=hf1_matrix[7,...])

            ch9 = f.create_group("channel_9")
            ch9.create_dataset("images", data=hf1_matrix[8,...])

            ch10 = f.create_group("channel_10")
            ch10.create_dataset("images", data=hf1_matrix[9,...])

            ch11 = f.create_group("channel_11")
            ch11.create_dataset("images", data=hf1_matrix[10,...])

            ch12 = f.create_group("channel_12")
            ch12.create_dataset("images", data=hf1_matrix[11,...])

# Call function
if wanted_function == "merge":
    print("Merging datasets")
    merge_datasets(file1_path, file2_path, save_path, output_name)
elif wanted_function == "split":
    print("Splitting dataset")
    split_train_test(hdf_file, save_path, test_amount, val_amount, output_name)
elif wanted_function == "fill_to_size":
    print("Creating new shape for file")
    fill_to_size(hdf_file, save_path, output_name)
else:
    print("No function chosen. Closing")

