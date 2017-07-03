#!/usr/bin/env python

"""
Usage:
    preprocessing.py <image_dir> <class_def>  [options]

Options:
    -o --output=output.npz                  Output file [default: output.npz].
    --test-split=0.2                        Test dataset ratio [default: 0.2].
    --save-split=True                       Whether to save splitting datasets [default: True].
"""


import numpy as np
import scipy as sp
from scipy.ndimage import imread
import random
import collections

from glob import glob
from docopt import docopt
import re


if __name__ == '__main__':
    # parse command options
    argv = docopt(__doc__)
    image_dir = argv['<image_dir>']
    class_def = argv['<class_def>']
    test_split = float(argv['--test-split'])
    output_file = argv['--output']

    # get classes
    classes = []
    with open(class_def, 'r') as f:
        lines = f.readlines()
    for line in lines:
        classes.append(re.sub('\n', '', line))

    file_list = glob('%s/*/*' % image_dir)

    # dataset splitting
    class_strs = []
    for f in file_list:
        class_strs.append(f.split('/')[-2])

    counter=collections.Counter(class_strs)
    class_strs = np.array(class_strs)
    train_files = []
    for k,v in counter.items():
        indices = np.where(class_strs == k)[0]
        nb_train = int(v * (1 - test_split))
        train_indices = random.sample(indices, k=nb_train)
        train_files += np.array(file_list)[train_indices].tolist()
    test_files = list(set(file_list) - set(train_files))

    images = []
    labels = []
    crop_flags = []
    test_flags = []
    for i,f in enumerate(train_files):
        print(i, f)
        img = imread(f)
        class_str = f.split('/')[-2]
        # image size >=  100x100
        if img.shape[0] >= 100 and\
            img.shape[1] >= 100:
            maybe_expanded_img = img
        # small image
        elif min(img.shape) < 100:
            if 50 <= img.shape[0] < 100:
                maybe_expanded_img = np.zeros((100, img.shape[1]))
                maybe_expanded_img[0:img.shape[0],:] = img.copy()
                maybe_expanded_img[img.shape[0]:,:] = img.copy()[0:100-img.shape[0],:]
            img = maybe_expanded_img
            if 50 <= img.shape[1] < 100:
                maybe_expanded_img = np.zeros((img.shape[0], 100))
                maybe_expanded_img[:,0:img.shape[1]] = img.copy()
                maybe_expanded_img[:,img.shape[1]:] = img.copy()[:,0:100-img.shape[1]]

        # make 100x100 crops
        img = maybe_expanded_img
        # standard image
        if img.shape[0] == img.shape[1] == 100:
            images.append(img)
            labels.append(classes.index(class_str))
            crop_flags.append(0)
        # big image
        else:
            n0 = int(np.ceil(img.shape[0] / 100.))
            n1 = int(np.ceil(img.shape[1] / 100.))
            n_crop = 0
            for nx in range(n0):
                if nx == (n0 - 1):
                    x0 = img.shape[0] - 100
                    x1 = img.shape[0]
                else:
                    x0 = nx * 100
                    x1 = (nx+1) * 100
                for ny in range(n1):
                    if ny == (n1 - 1):
                        y0 = img.shape[1] - 100
                        y1 = img.shape[1]
                    else:
                        y0 = ny * 100
                        y1 = (ny+1) * 100
                    crop = img[x0:x1, y0:y1]
                    images.append(crop)
                    labels.append(classes.index(class_str))
                    crop_flags.append(1)

    labels = np.array(labels, dtype=np.int16)
    crop_flags = np.array(crop_flags, dtype=np.int16)
    images = np.asarray(images)


    np.savez(output_file, images=images,
                          labels=labels,
                          crop_flags=crop_flags)

    if argv['--save-split'] == 'True':
        with open('train.txt', 'w') as f:
            for i in range(len(train_files)):
                f.write('%s\n' % train_files[i])   

        with open('test.txt', 'w') as f:
            for i in range(len(test_files)):
                f.write('%s\n' % test_files[i])