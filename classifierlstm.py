
import tensorflow as tf

from classifierbase import ClassifierBase

class Classifier(ClassifierBase):
    def __init__(self, args):
        ClassifierBase.__init__(self,args,'lstm-meanpool')

    def _forward(self, inputs, is_train):
        batch_x,len_x=inputs
        batch_size=tf.shape(batch_x)[0]
        n_emb=100
        vocab_size=100
        n_hidden=128
        num_steps=256
        n_class=2

        embedding = tf.get_variable("embedding", [vocab_size, n_emb], dtype=tf.float32)
        batch_emb = tf.nn.embedding_lookup(embedding, batch_x)

        with tf.variable_scope('lstm1') as scope:
            cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
            outputs,_ = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32,sequence_length=len_x,inputs=batch_emb)
        with tf.variable_scope('lstm2') as scope:
            cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, state_is_tuple=True)
            outputs,_ = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32,sequence_length=len_x,inputs=outputs)
        output=tf.reduce_sum(outputs,axis=1)

        softmax_w = tf.get_variable("softmax_w", [n_hidden, n_class], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [n_class], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b

        probs=tf.nn.softmax(logits)

        tf.get_variable_scope().reuse_variables()
        return logits,probs
