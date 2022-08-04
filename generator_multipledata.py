#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generator that takes in .h5-files made from cif-files with cifconvert.
Goes through all data files in folder, make sure the folder only contains what you want to train with. 
The searchstring option is for train, validation or test.
No IDEAS generated masks are used. 
Output is numpy array. 
"""

from normalize_background import normalize_background
import h5py
import numpy as np
import os
import random
import re

random.seed(42)

def naturalsorting(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def normalize(im, imin=0, imax=1):
   im = im.astype('float')
   if np.max(im) - np.min(im) == 0:
       return im
   else:
       return (im-np.min(im))*((imax-imin)/(np.max(im)-np.min(im))) + imin

def generator_multipledata(hdf5_directory, batch_size, searchstring, wanted_y_ch, normalized_background=False, quantile=0.1):
    while True:
        processed_files = []
        for fname in os.listdir(hdf5_directory):
            if searchstring in fname and fname not in processed_files: 
                with h5py.File(hdf5_directory + os.sep + fname, "r") as h5_file:
                    counter = 0
                    while counter < (len(h5_file["channel_1/images"]) - batch_size):
                        chans = list(h5_file.keys())
                        chans = naturalsorting(chans)
                        shape = tuple([len(chans)] + list(h5_file["channel_1/images"][:batch_size].shape))
                        image_matrix = np.empty(shape=shape, dtype=np.float32)
                        for i, chan in enumerate(chans):
                            ims = h5_file[chan]["images"][counter:(counter+batch_size)]
                            image_matrix[i,:,:,:] = ims
                        x = image_matrix[(0,8,11),...]
                        normalized_x = np.zeros((x.shape))
                        for channel in range(len(x)): 
                            index = 0
                            for image in x[channel,...]:
                                if index >= batch_size:
                                    index = 0
                                image = normalize(image, imin=0, imax=1)
                                normalized_x[channel,index,:,:] = image
                                index += 1
                        normalized_x = np.moveaxis(normalized_x,0,3)

                        y = image_matrix[wanted_y_ch,...]
                        normalized_y = np.zeros((y.shape))
                        index = 0
                        for image in y:
                            if index >= batch_size:
                                index = 0
                            image = normalize(image, imin=0, imax=1)
                            normalized_y[index,:,:] = image
                            index += 1
                        normalized_y = normalized_y.reshape(-1,128,128,1)

                        if normalized_background == True:
                            for i, image in enumerate(normalized_y):
                                normalized_y[i,...] = normalize_background(image, quantile=quantile)
                        counter += batch_size
                        yield (normalized_x, normalized_y)
                processed_files.append(fname)
            
            else:
                print("This file is not a wanted file", fname)
                continue

        print("Finished for this epoch. Went through files:", processed_files)
