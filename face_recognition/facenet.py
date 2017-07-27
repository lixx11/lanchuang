import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer

import numpy as np


class FaceNet(object):
    """docstring for FaceNet"""
    def __init__(self, model_file):
        self.model_file = model_file
        self.load_model()

    def load_model(self):
        graph_def = graph_pb2.GraphDef()

        with open(self.model_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        importer.import_graph_def(graph_def, name='')

    def calc_embedding(self, data, sess):
        g = tf.get_default_graph()
        images_placeholder = g.get_tensor_by_name("input:0")
        embeddings = g.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = g.get_tensor_by_name("phase_train:0")

        embedding = sess.run([embeddings], 
            feed_dict={images_placeholder: np.array(data), 
                       phase_train_placeholder: False })[0]
        return embedding



if __name__ == '__main__':
    model_file = 'model_check_point/20170512-110547.pb'
    facenet = FaceNet(model_file)

    data = np.random.rand(1, 160, 160, 3)

    with tf.Session() as sess:
        embedding = facenet.calc_embedding(data, sess)
