#!/usr/bin/env python

import tensorflow as tf 
import numpy as np
from scipy.ndimage import imread

import os
import sys
import re
import time

import cnn_model


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_file', None,
                           """Path of data file to evaluate.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                           """Number of samples in a batch""")
tf.app.flags.DEFINE_bool('center_only', True,
                           """Only evaluate central crop.""")
tf.app.flags.DEFINE_integer('nb_crop_per_image', 5,
                           """Number of crops for each image""")
tf.app.flags.DEFINE_string('ckpt_file', None,
                            """Global step of ckpt file.""")


def eval():
    # parameters and options
    data_file = FLAGS.data_file
    crop_size = FLAGS.crop_size
    batch_size = FLAGS.batch_size
    nb_crop_per_image = FLAGS.nb_crop_per_image

    nb_success = 0
    nb_failure = 0

    # load data
    with open(data_file) as f:
        img_files = f.readlines()
    print('%d images to be evaluated' % len(img_files))

    # get classes
    class_strs = []
    for f in img_files:
        class_strs.append(f.split('/')[-2])
    classes = np.unique(class_strs).tolist()
    classes.sort()


    with tf.Graph().as_default():
        if FLAGS.use_fp16:
            FP = tf.float16
        else:
            FP = tf.float32
        images = tf.placeholder(FP, 
                shape=[batch_size, crop_size, crop_size, 1], 
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

            begin = time.time()
            time_to_read_img = 0.
            time_to_eval_img = 0.
            eval_times = []
            for i in range(len(img_files)):
                img_file = re.sub('\n', '', img_files[i])
                print('processing %s' % img_file)
                t0 = time.time()
                image = imread(img_file)
                t1 = time.time()
                time_to_read_img += (t1 - t0)

                if not FLAGS.center_only:
                    x_range = image.shape[0] - crop_size
                    y_range = image.shape[1] - crop_size
                    xs = np.random.choice(x_range, nb_crop_per_image)
                    ys = np.random.choice(y_range, nb_crop_per_image)
                    prob = np.zeros(len(classes))
                    for (x, y) in zip(xs, ys):
                        crop = image[x:x+crop_size, y:y+crop_size]
                        crop = crop.reshape((1, crop_size, crop_size, 1))   

                        logits_value = sess.run(logits,
                            feed_dict={images: crop})
                        exp_logits = np.exp(logits_value)
                        prob_ = exp_logits.T / np.sum(exp_logits, axis=1)
                        prob += prob_.reshape(-1)
                    prob /= float(nb_crop_per_image)
                else:
                    x = int((image.shape[0] - crop_size) / 2.)
                    y = int((image.shape[1] - crop_size) / 2.)
                    crop = image[x:x+crop_size, y:y+crop_size]
                    crop = crop.reshape((1, crop_size, crop_size, 1))   

                    logits_value = sess.run(logits,
                        feed_dict={images: crop})
                    exp_logits = np.exp(logits_value)
                    prob = exp_logits.T / np.sum(exp_logits, axis=1)
                    prob = prob.reshape(-1)

                pred_label = classes[np.argmax(prob)]

                true_label = img_file.split('/')[-2]
                t2 = time.time()
                eval_times.append(t2 - t1)
                time_to_eval_img += (t2 - t1)
                if true_label == pred_label:
                    print('success(%.3f)' % prob.max())
                    nb_success += 1
                else:
                    print('failure(%.3f): %s -> %s' % 
                        (prob.max(), true_label, pred_label))
                    nb_failure += 1
            end = time.time()
            print('time elapsed %.3f sec, %.3f(read img) + %.3f(eval img)' 
                % ((end - begin), time_to_read_img, time_to_eval_img))
            print('%d images processed with %d success and %d failure' %
                (len(img_files), nb_success, nb_failure))
            np.savetxt('eval_times.log', eval_times, fmt='%.4f')


def main(argv=None):  # pylint: disable=unused-argument
    eval()


if __name__ == '__main__':
    tf.app.run()