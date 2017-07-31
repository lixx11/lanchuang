#!/usr/bin/env python


import tensorflow as tf 

import numpy as np

import time
from datetime import datetime
from glob import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('data_file', None,
                           """Path of training data file""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                           """Number of samples in a batch""")
tf.app.flags.DEFINE_string('train_dir', 'train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_integer('decay_steps', 10,
                            """Decay step for learning rate.""")
tf.app.flags.DEFINE_float('decay_factor', 0.99,
                            """Decay factor for learning rate.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 5,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_boolean('shuffle', True,
                            """Whether to shuffle the batch.""")
tf.app.flags.DEFINE_boolean('load_ckpt', False,
                            """Whether to load checkpoint file.""")
tf.app.flags.DEFINE_integer('ckpt_step', 0,
                            """Global step of ckpt file.""")

import cnn_model
from cnn_input import CNNTrainInput


def train():
    batch_size = FLAGS.batch_size
    crop_size = FLAGS.crop_size

    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # input
        train_input = CNNTrainInput(FLAGS.data_file)

        images = tf.placeholder(tf.float32, 
                shape=[batch_size, crop_size, crop_size, 1], 
                name='image')
        labels = tf.placeholder(tf.int64, shape=[batch_size], name='label')
        # Display the training images in the visualizer.
        
        image_slices = tf.slice(images, 
            [0,0,0,0], 
            [int(batch_size), crop_size, crop_size, 1],
            name='central_slice')
        tf.summary.image('image_slices', image_slices, max_outputs=10)

        # inference
        logits = cnn_model.inference(images)

        # calculate accuracy and error rate
        prediction = tf.argmax(logits,1)
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        error_rate = 1 - accuracy
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('error_rate', error_rate)

        # train to minimize loss
        loss = cnn_model.loss(logits, labels)
        lr = tf.placeholder(tf.float64, name='leaning_rate')
        tf.summary.scalar('learning_rate', lr)
        train_op = cnn_model.train(loss, lr, global_step)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # run graph in session
        with tf.Session() as sess:
            init = tf.global_variables_initializer() # create an operation initializes all the variables
            sess.run(init)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('%s' % FLAGS.train_dir, sess.graph)
            
            if FLAGS.load_ckpt:
                ckpt_file = '%s/model.ckpt-%d' % \
                    (FLAGS.train_dir, FLAGS.ckpt_step)
                print('restore sess with %s' % ckpt_file)
                saver.restore(sess, ckpt_file)

            start = time.time()
            for step in range(FLAGS.max_steps):
                batch_images, batch_labels = train_input.next_batch(batch_size=batch_size)
                if FLAGS.shuffle:
                    idx = np.arange(batch_size)
                    np.random.shuffle(idx)
                    batch_images = batch_images[idx]
                    batch_labels = batch_labels[idx]
                lr_value = FLAGS.learning_rate * pow(FLAGS.decay_factor, 
                    (step / FLAGS.decay_steps))
                _, err, g_step, loss_value, summary = sess.run(
                    [train_op, error_rate, global_step, loss, merged], 
                    feed_dict={
                        labels: batch_labels,
                        images: batch_images,
                        lr: lr_value,
                    })

                if step % FLAGS.log_frequency == 0 and step != 0:
                    end = time.time()
                    duration = end - start
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)
                    format_str = ('%s: step %d, loss = %.2f, err = %.4f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print (format_str % (datetime.now(), g_step, loss_value, err, 
                               examples_per_sec, sec_per_batch))
                    writer.add_summary(summary, g_step)
                    # Save the variables to disk.
                    saver.save(sess, '%s/model.ckpt' % FLAGS.train_dir,  global_step=g_step)
                    start = end


def main(argv=None):  # pylint: disable=unused-argument
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()