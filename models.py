import os
import tensorflow as tf
import config

class Model:
    def __init__(self, training=True):
        self.training = training

    def _create_placeholder(self):
        with tf.name_scope('data'):
            self.signals = tf.placeholder(tf.float32, [None, config.max_seq_length], name="signals_placeholder")
            self.labels = tf.placeholder(tf.int64, [None, config.max_base_length], name="label_placeholder")
            self.sig_length = tf.placeholder(tf.int64, [None], name='sig_length_placeholder')
            self.base_length = tf.placeholder(tf.int64, [None], name='base_length_placeholder')
    
    def _inference(self):
        pass

    def _loss(self):
        pass

    def _train_op(self):
        pass

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        self._loss()
        self._train_op()

class Baseline(Model):
    def _encode(self):
        

    def _inference(self)
        _encode()
        _decode()