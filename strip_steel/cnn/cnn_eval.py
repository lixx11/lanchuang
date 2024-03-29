#!/usr/bin/env python

import tensorflow as tf 

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
tf.app.flags.DEFINE_string('class_def', None,
                            """Class definition file.""")
tf.app.flags.DEFINE_bool('save_pb', False,
                           """Save protobuf.""")
tf.app.flags.DEFINE_string('pb_name', 'output.pb',
                            """Filename of protobuf.""")

import cnn_model

import numpy as np
from scipy.ndimage import imread

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import re
import time


def eval():
    # parameters and options
    data_file = FLAGS.data_file
    crop_size = FLAGS.crop_size
    class_def = FLAGS.class_def
    batch_size = FLAGS.batch_size
    nb_crop_per_image = FLAGS.nb_crop_per_image

    nb_success = 0
    nb_failure = 0

    # load data
    with open(data_file) as f:
        img_files = f.readlines()
    print('%d images to be evaluated' % len(img_files))

    # get classes
    classes = []
    with open(class_def, 'r') as f:
        lines = f.readlines()
    for line in lines:
        classes.append(re.sub('\n', '', line))


    with tf.Graph().as_default():
        images = tf.placeholder(tf.float32, 
                shape=[batch_size, crop_size, crop_size, 1], 
                name='image')

        # inference
        logits = cnn_model.inference(images)
        if FLAGS.save_pb:
            g = tf.get_default_graph()
            with open(FLAGS.pb_name, 'wb') as f:
                f.write(g.as_graph_def().SerializeToString())
            print('write graphdef to %s' % FLAGS.pb_name)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # run graph in session
        with tf.Session() as sess:
            init = tf.global_variables_initializer()

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
            print('time elapsed %.3f sec, %.3f(read) + %.3f(eval)' 
                % ((end - begin), time_to_read_img, time_to_eval_img))
            print('%d images processed with %d success and %d failure' %
                (len(img_files), nb_success, nb_failure))


def main(argv=None):  # pylint: disable=unused-argument
    eval()


if __name__ == '__main__':
    tf.app.run()