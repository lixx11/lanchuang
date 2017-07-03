#!/usr/bin/env python

import numpy as np 
from scipy.ndimage.interpolation import map_coordinates
import random
from math import pi, sin, cos

import time
from glob import glob
import os


ROTATION_RANGE = [0, 360]
STRENTCH_RANGE = [0.8, 1.2]
NOISE_RANGE = [0.8, 1.2]

class CNNTrainInput(object):
    """docstring for CNNTrainInput"""
    def __init__(self, 
                 data_file,
                 crop_size=64):
        self.data_file = data_file
        self.crop_size = crop_size

        data = np.load(self.data_file)
        self.images = data['images']
        self.labels = data['labels']
        self.classes = np.unique(self.labels)
        self.nb_classes = len(self.classes)

    def next_batch(self, 
                   batch_size=128,
                   isoprob=True):
        batch_images = []
        batch_labels = []

        sample_indices = []
        if isoprob:
            nb_samples_per_class = int(np.ceil(
                float(batch_size) / self.nb_classes))
            for c in self.classes:
                indices = np.where(self.labels == c)[0]
                rd_indices = np.random.choice(indices, nb_samples_per_class)
                sample_indices += rd_indices.tolist()
        else:
            rd_indices = np.random.choice(np.arange(len(self.labels)), batch_size)
        # collect samples
        for i in range(batch_size):
            idx = sample_indices[i]
            image = self.images[idx].copy()
            img_size = image.shape[0]
            crop_size = self.crop_size
            x0, y0 = np.random.choice(np.arange(img_size-crop_size), 2)
            crop = image[x0:x0+crop_size, y0:y0+crop_size]
            batch_images.append(crop)
            batch_labels.append(self.labels[idx])

        batch_images = np.asarray(batch_images)
        batch_images = batch_images.reshape((batch_size, 
            self.crop_size, self.crop_size, 1))
        batch_labels = np.asarray(batch_labels)
        return batch_images, batch_labels
        

if __name__ == '__main__':
    data_file = '../output.npz'
    train_input = CNNTrainInput(data_file, crop_size=64)
    images, labels = train_input.next_batch(batch_size=128)