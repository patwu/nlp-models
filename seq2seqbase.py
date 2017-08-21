
import tensorflow as tf
import numpy as np

from networkbase import NetworkBase

class Seq2SeqBase(NetworkBase):

    def _loss(self, logits, targets):
        d,md,ld=targets
        d=d[:,1:]
        md=md[:,1:]
        ld=tf.subtract(ld,1)
        shape=tf.shape(d)
        d=tf.reshape(d,[shape[0]*shape[1]])
        md=tf.reshape(md,[shape[0]*shape[1]])
        self.debug=logits,d
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=d, name='cross_entropy_per_example')
        ce=tf.multiply(ce,md)
        ce=tf.reshape(ce,[shape[0],shape[1]])
        cem=tf.div(tf.reduce_sum(ce,axis=1),tf.cast(ld,dtype=tf.float32))
        cem = tf.reduce_mean(cem, name='cross_entropy')
        return cem

    def _decode_step(self,inputs):
        pass

    def _decode_all(self,inputs):
        pass

    def _encode(self,inputs):
        pass

    def build_input(self):
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                src=self.src=tf.placeholder(tf.int64,name='src')
                s_mask=self.s_mask=tf.placeholder(tf.float32,name='s_mask')
                s_len=self.s_len=tf.placeholder(tf.int64,name='s_len')
                dest=self.dest=tf.placeholder(tf.int64,name='dest')
                d_mask=self.d_mask=tf.placeholder(tf.float32,name='d_mask')
                d_len=self.d_len=tf.placeholder(tf.int64,name='d_len')
                key=self.key=tf.placeholder(tf.int64,name='key')

                global_step = self.global_step= tf.Variable(0, name='global_step', trainable=False)
                enc_shape=[self.args.enc_max_len]
                dec_shape=[self.args.dec_max_len]

                if self.args.shuffle_input==True:
                    sample_queue=tf.RandomShuffleQueue(capacity=4096, min_after_dequeue=1024, 
                        dtypes=[tf.int64,tf.float32,tf.int64,tf.int64,tf.float32,tf.int64], 
                        shapes=[enc_shape,enc_shape,[],dec_shape,dec_shape,[]]
                    )
                else:
                    sample_queue=tf.FIFOQueue(capacity=1024, 
                        dtypes=[tf.int64,tf.float32,tf.int64,tf.int64,tf.float32,tf.int64], 
                        shapes=[enc_shape,enc_shape,[],dec_shape,dec_shape,[]]
                    )
                self.sample_queue=sample_queue
                self.feed_step=sample_queue.enqueue((src,s_mask,s_len,dest,d_mask,d_len,key))


    def build_model(self):
        with self.graph.as_default():
            is_train=self.is_train=tf.placeholder(tf.bool,name='is_train')
            batch_size=self.args.batch_size
            n_gpu=len(self.args.gpu_list.split(','))

            if self.args.is_train==False:
                e=self.e=tf.placeholder(tf.int64,shape=[batch_size,self.args.enc_max_len],name='e')
                le=self.le=tf.placeholder(tf.int64,shape=[batch_size],name='le')

                one_code=self.one_code=tf.placeholder(tf.int64,shape=[batch_size,1],name='one_code')
                state=self.state_feed=tf.placeholder(tf.float32,shape=[2,batch_size,128],name='state')
                with tf.device('/gpu:0'):
                    self.encode_state=self._encode([e,le])  
                    self.prob,self.state_ret=self._decode_step([state,one_code])  

            else:
                with tf.device('/cpu:0'):
                    if self.args.start_lr is None:
                        opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
                    else:
                        lr = tf.train.exponential_decay(self.args.start_lr, self.global_step, 1500000, 0.5, staircase=True)
                        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
                
                e,me,le,d,md,ld= self.sample_queue.dequeue_many(batch_size)
                
                batch_e=tf.split(e,num_or_size_splits=n_gpu,axis=0)
                batch_me=tf.split(me,num_or_size_splits=n_gpu,axis=0)
                batch_le=tf.split(le,num_or_size_splits=n_gpu,axis=0)

                batch_d=tf.split(d,num_or_size_splits=n_gpu,axis=0)
                batch_md=tf.split(md,num_or_size_splits=n_gpu,axis=0)
                batch_ld=tf.split(ld,num_or_size_splits=n_gpu,axis=0)

                tower_grads=[]
                tower_loss=[]
                tower_state=[]
                for i in range(n_gpu):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                            state=self._encode([batch_e[i],batch_le[i]])
                            logits,prob = self._decode_all([state,batch_d[i][:,:-1],tf.subtract(batch_ld[i],1)])
                            tf.get_variable_scope().reuse_variables()

                            loss = self._loss(logits, [batch_d[i], batch_md[i], batch_ld[i]])
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
                            tower_loss.append(loss)
                grads = self._average_gradients(tower_grads)
                apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
                self.train_step = apply_gradient_op
                self.loss=tower_loss[0]

                #summary
                if not self.args.start_lr is None:
                    tf.summary.scalar('lr',lr)
                tf.summary.scalar('%s-seq2seq-loss'%self.name,self.loss)
                self.summary_step=tf.summary.merge_all()

            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    def push_pipline(self,src,s_mask,s_len,dest,d_mask,d_len,key):
        feed = {self.src:src,self.s_mask:s_mask,self.s_len:s_len,
                self.dest:dest,self.d_mask:d_mask,self.d_len:d_len,
                self.key:key}
        self.sess.run([self.feed_step], feed_dict=feed)

    def greedy_translate(self,src,s_len):
        while(len(src)!=self.args.enc_max_len):
            src.append(0)
        feed = {self.e:[src],self.le:[s_len]}
        state=self.sess.run([self.encode_state],feed_dict=feed)[0]
        sentence=[1]
        while not sentence[-1]==2:
            feed={self.state_feed:state,self.one_code:[[sentence[-1]]]}
            state,prob,debug=self.sess.run([self.state_ret,self.prob,self.debug],feed_dict=feed)
            sentence.append(np.argmax(prob))
        return sentence

