
# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Important functions for DeepIFC.
"""
import matplotlib.pyplot as plt
import numpy as np
import re

def naturalsorting(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    number_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=number_key)

def normalize_background(image, quantile=0.6):
    image2 = image[image>0]
    quantilevalue = np.quantile(image2, quantile)
    image -= quantilevalue
    clipped_image = np.maximum(0, image)
    return clipped_image

def normalize(im, imin=0, imax=1):
    im = im.astype('float')
    if np.max(im) - np.min(im) == 0:
        return im
    else:
        return (im-np.min(im))*((imax-imin)/(np.max(im)-np.min(im))) + imin

def normalize_list(x):
    return (x - x.min()) / (np.ptp(x))
    return norm

def func(a, b):
    return any([i in b for i in a])

def percentage(part, whole):
    return float(part)/float(whole) * 100