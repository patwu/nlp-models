import sys
import numpy as np
import tensorflow as tf
import argparse
import os
import threading
import time

class NetworkBase(object):
    def __init__(self, args, name):
        self.args=args
        self.name=name
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        self.graph = tf.Graph()
        self.sess=tf.Session(graph=self.graph, config=config)

    def _forwoard(self,inputs,is_train):
        pass

    def _loss(self, xs, ys):
        pass

    def _average_gradients(self,tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads,axis=0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            if not self.args.grad_clip is None:
                grad = tf.clip_by_norm(grad, clip_norm=self.args.grad_clip)
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def build_input(self):
        pass 

    def build_model(self):
        pass

    def save_model(self, path=None):
        if path is None:
            path=self.args.model_path
        self.saver.save(self.sess, os.path.join(path,'model.ckpt'), global_step=self.global_step)

    def load_model(self, path=None):
        if path is None:
            path=self.args.model_path
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            print "Load model %s" % (ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            print "No model."
            return False

