#!/usr/bin/env python

import tensorflow as tf 
import numpy as np
from scipy.ndimage import imread

import os
import sys
import re

import cnn_model


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_file', None,
                           """Path of data file to evaluate, single path
                           or multiple path in txt file.
                           """)
tf.app.flags.DEFINE_integer('batch_size', 1,
                           """Number of samples in a batch""")
tf.app.flags.DEFINE_integer('nb_crop_per_image', 5,
                           """Number of crops for each image""")
tf.app.flags.DEFINE_string('ckpt_file', None,
                            """Global step of ckpt file.""")
tf.app.flags.DEFINE_string('class_def', None,
                            """Class definition file.""")



def eval():
    # parameters and options
    data_file = FLAGS.data_file
    crop_size = FLAGS.crop_size
    class_def = FLAGS.class_def
    nb_crop_per_image = FLAGS.nb_crop_per_image

    nb_success = 0
    nb_failure = 0

    # get classes
    classes = []
    with open(class_def, 'r') as f:
        lines = f.readlines()
    for line in lines:
        classes.append(re.sub('\n', '', line))

    # load data
    _, ext = os.path.splitext(data_file)
    if ext == '.jpg':  # single jpg to evaluate
        img_files = [data_file]
    elif ext == '.txt':  # multiple image
        with open(data_file) as f:
            img_files = f.readlines()
    else:
        print('Unsupported data file format: %s' % ext)
        print('Please provide single image in jpg format or'
            'multiple images in txt format.')
        sys.exit()
    print('%d images to be evaluated' % len(img_files))

    with tf.Graph().as_default():
        if FLAGS.use_fp16:
            FP = tf.float16
        else:
            FP = tf.float32
        images = tf.placeholder(FP, 
                shape=[1, crop_size, crop_size, 1], 
                name='image')

        # inference
        logits = cnn_model.inference(images)
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # run graph in session
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, FLAGS.ckpt_file)

            for i in range(len(img_files)):
                img_file = re.sub('\n', '', img_files[i])
                print('processing %s' % img_file)
                image = imread(img_file)
                x_range = image.shape[0] - crop_size
                y_range = image.shape[1] - crop_size
                xs = np.random.choice(x_range, nb_crop_per_image)
                ys = np.random.choice(y_range, nb_crop_per_image)
                probs = np.zeros(len(classes))
                for (x, y) in zip(xs, ys):
                    crop = image[x:x+crop_size, y:y+crop_size]
                    crop = crop.reshape((1, crop_size, crop_size, 1))
                    logits_value = sess.run(logits,
                        feed_dict={images: crop})
                    exp_logits = np.exp(logits_value)
                    prob = exp_logits.T / np.sum(exp_logits, axis=1)
                    probs += prob.reshape(-1)
                true_label = img_file.split('/')[-2]
                pred_label = classes[np.argmax(probs)]

                if true_label == pred_label:
                    print('success')
                    nb_success += 1
                else:
                    print('failure: %s -> %s' % 
                        (true_label, pred_label))
                    nb_failure += 1

            print('%d images processed with %d success and %d failure' %
                (len(img_files), nb_success, nb_failure))


def main(argv=None):  # pylint: disable=unused-argument
    eval()


if __name__ == '__main__':
    tf.app.run()