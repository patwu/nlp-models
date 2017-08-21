
import tensorflow as tf

from seq2seqbase import Seq2SeqBase

class seq2seq(Seq2SeqBase):

    def __init__(self,args):
        Seq2SeqBase.__init__(self,args,'simple')
        self.n_hidden=128
        self.n_embedding=100

        self.n_encode_voc=3500
        self.n_decode_voc=30


    def _encode(self,inputs):
        batch_e, batch_le=inputs
        enc_emb = tf.get_variable("encoding-embedding", [self.n_encode_voc, self.n_embedding], dtype=tf.float32)
        batch_e = tf.nn.embedding_lookup(enc_emb,batch_e)
        with tf.variable_scope('encoder') as scope:
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=0.0, state_is_tuple=True)
            _,state = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32,sequence_length=batch_le,inputs=batch_e)
        return state

    def _decode_step(self,inputs):
        state,one_code=inputs
        state=tf.contrib.rnn.LSTMStateTuple(state[0,:,:],state[1,:,:])
        self.debug=state
        dec_emb = tf.get_variable("decoding-embedding", [self.n_decode_voc, self.n_embedding], dtype=tf.float32)
        one_code = tf.nn.embedding_lookup(dec_emb,one_code)
        l=tf.ones(shape=[self.args.batch_size])

        with tf.variable_scope('decoder') as scope:
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=0.0, state_is_tuple=True)
            outputs,state = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32,sequence_length=l,inputs=one_code,initial_state=state)
        logits=tf.contrib.layers.fully_connected(outputs,self.n_decode_voc,activation_fn=None)
        probs=tf.nn.softmax(logits)
        return probs,state
       


    def _decode_all(self,inputs):
        state,batch_d,batch_ld=inputs
        dec_emb = tf.get_variable("decoding-embedding", [self.n_decode_voc, self.n_embedding], dtype=tf.float32)
        batch_d = tf.nn.embedding_lookup(dec_emb,batch_d)

        with tf.variable_scope('decoder') as scope:
            cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=0.0, state_is_tuple=True)
            outputs,_ = tf.nn.dynamic_rnn(cell=cell,dtype=tf.float32,sequence_length=batch_ld,inputs=batch_d,initial_state=state)
        shape=tf.shape(outputs)
        outputs=tf.reshape(outputs,(shape[0]*(self.args.dec_max_len-1),self.n_hidden))

        logits=tf.contrib.layers.fully_connected(outputs,self.n_decode_voc,activation_fn=None)
        probs=tf.nn.softmax(logits)
        return logits,probs

