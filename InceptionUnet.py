#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modified from InceptionUnet:
https://github.com/danielenricocahall/Keras-UNet/blob/master/UNet/createtInceptionUNet.py
"""

from keras.models import Model, Input
from keras.layers import Convolution2D, Activation, BatchNormalization,MaxPooling2D, concatenate
from keras.layers.convolutional import UpSampling2D
from Inception import InceptionModule


def createInceptionUnet(pretrained_weights = None,
                        input_shape = (128,128,3), 
                        n_labels = 1,
                        numFilters = 8,
                        output_mode="sigmoid"):

    inputs = Input(input_shape)
    conv1 = InceptionModule(inputs, numFilters)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(conv1)
    conv2 = InceptionModule(pool1, 2*numFilters)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(conv2)
    conv3 = InceptionModule(pool2, 4*numFilters)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(conv3)
    conv4 = InceptionModule(pool3, 8*numFilters)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool4")(conv4)
    conv5 = InceptionModule(pool4,16*numFilters)
    pool5 = MaxPooling2D(pool_size=(2,2), name="pool5")(conv5)
    conv6 = InceptionModule(pool5, 16*numFilters) 
    pool6 = MaxPooling2D(pool_size=(2,2), name="pool6")(conv6)
    conv7 = InceptionModule(pool6, 16*numFilters) 
    pool7 = MaxPooling2D(pool_size=(2,2), name="pool7")(conv7)
    conv8 = InceptionModule(pool7, 16*numFilters) 
    pool8 = MaxPooling2D(pool_size=(1,1), name="pool8")(conv8)
    conv9 = InceptionModule(pool8, 16*numFilters) # bottleneck of the model
    up5 = UpSampling2D(size=(2,2), name="added_upsampling5")(conv9)
    up5 = InceptionModule(up5, 16*numFilters)
    up6 = UpSampling2D(size=(2,2), name="added_upsampling6")(up5)
    up6 = InceptionModule(up6, 16*numFilters)
    up7 = UpSampling2D(size=(2,2), name="added_upsampling7")(up6)
    up7 = InceptionModule(up7, 16*numFilters)
    merge5 = concatenate([conv5,up7],axis=3)
    up6 = UpSampling2D(size=(2,2), name="upsampling_6")(merge5)
    up6 = InceptionModule(up6, 8*numFilters)
    merge6 = concatenate([conv4,up6],axis=3)
    up7 = UpSampling2D(size=(2,2))(merge6)
    up7 = InceptionModule(up7, 4*numFilters)
    merge7 = concatenate([conv3,up7],axis=3)
    up8 = UpSampling2D(size=(2,2))(merge7)
    up8 = InceptionModule(up8, 2*numFilters)
    merge8 = concatenate([conv2,up8],axis=3)
    up9 = UpSampling2D(size=(2,2))(merge8)
    up9 = InceptionModule(up9, numFilters)
    merge9 = concatenate([conv1,up9],axis=3)
    conv10 = Convolution2D(n_labels, (1,1),  padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv10 = BatchNormalization()(conv10)
    outputs = Activation(output_mode)(conv10)
    model = Model(inputs, outputs) 
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)
    return model
