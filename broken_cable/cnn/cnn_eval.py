#!/usr/bin/env python


import tensorflow as tf 

FLAGS = tf.app.flags.FLAGS

import numpy as np
from math import pow

import cnn_model
from cnn_input import CNNEvalInput

import time
from datetime import datetime
from glob import glob


# Basic model parameters.
tf.app.flags.DEFINE_string('data_file', None,
                           """Path of data file for evaluation""")
tf.app.flags.DEFINE_string('ckpt_file', None,
                            """Global step of ckpt file.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                           """Number of samples in a batch""")


def eval(data_file=None, 
        crop_size=None, 
        ckpt_file=None,
        batch_size=128):
    with tf.Graph().as_default():
        if FLAGS.use_fp16:
            FP = tf.float16
        else:
            FP = tf.float32
        eval_input = CNNEvalInput(data_file,
                                crop_size=crop_size)
        images = tf.placeholder(FP, 
                shape=[batch_size, crop_size[0], crop_size[1], 1], 
                name='image')
        # inference
        logits = cnn_model.inference(images)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # run graph in session
        with tf.Session() as sess:
            init = tf.global_variables_initializer() # create an operation initializes all the variables
            sess.run(init)
            print('restore sess with %s' % ckpt_file)
            saver.restore(sess, ckpt_file)

            n_frames = eval_input.data.shape[0]
            probs = np.zeros((n_frames, 2))
            count = 0
            start = time.time()
            while True:
                batch_images, ending = eval_input.next_batch(batch_size=batch_size)
                logits_value = sess.run(logits, 
                    feed_dict={
                        images: batch_images,
                    })
                exp_logits = np.exp(logits_value)
                prob = exp_logits.T / np.sum(exp_logits, axis=1)

                if ending:
                    probs[count:] = prob.T[:n_frames-count]
                    break
                else:
                    probs[count:count+batch_size] = prob.T
                    count += batch_size
            end = time.time()
            print('time elapsed %.3f' % (end-start))
            return probs


def main(argv=None):  # pylint: disable=unused-argument
    data_file = FLAGS.data_file
    batch_size = FLAGS.batch_size
    ckpt_file = FLAGS.ckpt_file
    crop_size = tuple(map(int, FLAGS.crop_size.split('x')))
    eval(data_file=data_file, 
        crop_size=crop_size, 
        ckpt_file=ckpt_file,
        batch_size=batch_size)


if __name__ == '__main__':
    tf.app.run()