#!/usr/bin/env python

import numpy as np 
import random

import time
from glob import glob
import os


class CNNTrainInput(object):
    """docstring for CNNTrainInput"""
    def __init__(self, 
                 data_dir,
                 crop_size=(100, 270)):
        self.data_dir = data_dir
        self.crop_size = crop_size

        self.data_files = glob('%s/*train*.npz' % self.data_dir)

        self.macro_batch = None
        self.micro_batch = None

    def next_macro_batch(self,
                        nb_data_files=3,
                        keep=100):
        macro_batch = {'labels': None,
                       'frames': None}
        data_files_curr = np.random.choice(self.data_files,
            size=min(nb_data_files, len(self.data_files)),
            replace=False)
        for data_file in data_files_curr:
            data = np.load(data_file)
            if macro_batch['labels'] is None:
                macro_batch['labels'] = data['labels']
                macro_batch['frames'] = data['frames']
            else:
                macro_batch['labels'] = np.concatenate((macro_batch['labels'],
                    data['labels']))
                macro_batch['frames'] = np.concatenate((macro_batch['frames'],
                    data['frames']))
        if len(np.unique(macro_batch['labels'])) == 1:
            print('WARNING!!! Only 1 class in this macro batch')
        return macro_batch


    def next_micro_batch(self, 
                        batch_size=128, 
                        isoprob=True):
        if self.macro_batch is None:
            self.macro_batch = self.next_macro_batch()

        batch_images = []
        batch_labels = []
        classes = np.unique(self.macro_batch['labels'])

        sample_indices = []
        if isoprob:
            nb_samples_per_class = int(np.ceil(
                float(batch_size) / len(classes)))
            for c in classes:
                indices = np.where(self.macro_batch['labels'] == c)[0]
                rd_indices = np.random.choice(indices, nb_samples_per_class)
                sample_indices += rd_indices.tolist()
        else:
            rd_indices = np.random.choice(np.arange(len(self.labels)), batch_size)
        # collect samples
        for i in range(batch_size):
            idx = sample_indices[i]
            image = self.macro_batch['frames'][idx]
            img_size = image.shape
            crop_size = self.crop_size
            x0 = int(np.random.choice(np.arange(img_size[0]-crop_size[0]), 1))
            y0 = int(np.random.choice(np.arange(img_size[1]-crop_size[1]), 1))
            crop = image[x0:x0+crop_size[0], y0:y0+crop_size[1]]
            batch_images.append(crop)
            batch_labels.append(self.macro_batch['labels'][idx])

        batch_images = np.asarray(batch_images)
        batch_images = batch_images.reshape((batch_size, 
            self.crop_size[0], self.crop_size[1], 1))
        batch_labels = np.asarray(batch_labels)
        return batch_images, batch_labels
        

class CNNEvalInput(object):
    """docstring for CNNTrainInput"""
    def __init__(self, 
                 data_file,
                 crop_size=(100, 270)):
        self.data_file = data_file
        self.crop_size = crop_size
        self.data = np.load(self.data_file)['frames']
        self._frame_count = 0

    def next_batch(self, batch_size=128):
        batch_images = []
        ending = False

        # collect samples
        data = self.data
        n_batch = int(np.ceil(data.shape[0] / batch_size))
        _, sx, sy = data.shape
        cx, cy = self.crop_size
        x0 = int((sx - cx)/2)
        y0 = int((sy - cy)/2)
        _fc = self._frame_count
        batch_images = data[_fc:_fc+batch_size,
                            x0:x0+cx,
                            y0:y0+cy]
        if batch_images.shape[0] < batch_size:
            n_extra = batch_size - batch_images.shape[0]
            extra_images = np.ones((n_extra, cx, cy))
            batch_images = np.concatenate((batch_images, extra_images))
        batch_images = batch_images.reshape((batch_size, 
            self.crop_size[0], self.crop_size[1], 1))
        self._frame_count += batch_size
        if self._frame_count >= data.shape[0]:
            ending = True
        return batch_images, ending


if __name__ == '__main__':
    # data_dir = '..'
    # train_input = CNNTrainInput(data_dir, crop_size=(100, 270))
    # images, labels = train_input.next_micro_batch(batch_size=128)

    data_file = '/Users/lixuanxuan/Repository/lanchuang/broken_cable/20170705-09-train-00.npz'
    eval_input = CNNEvalInput(data_file, crop_size=(100, 270))
    batch_images, ending = eval_input.next_batch()