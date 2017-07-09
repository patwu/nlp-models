
import tensorflow as tf

from networkbase import NetworkBase

class ClassifierBase(NetworkBase):

    def _loss(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    def build_input(self, extract_func):
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                sentence = self.sentence = tf.placeholder(tf.string, name='sentence')
                label= self.label = tf.placeholder(tf.int64, name='label')
                key=self.key=tf.placeholder(tf.int64,name='key')
                global_step = self.global_step= tf.Variable(0, name='global_step', trainable=False)
                sample,mask = extract_func(sentence)
                if self.args.shuffle_input==True:
                    sample_queue=tf.RandomShuffleQueue(capacity=4096, min_after_dequeue=1024, dtypes=[tf.int32,tf.int32,tf.int64,tf.int64], shapes=[[256],[],[],[]])
                else:
                    sample_queue=tf.FIFOQueue(capacity=1024, dtypes=[tf.int32,tf.int32,tf.int64,tf.int64], shapes=[[256],[],[],[]])
                self.sample_queue=sample_queue
                self.feed_step=sample_queue.enqueue((sample,mask[0],label,key))

    def build_model(self):
        with self.graph.as_default():
            is_train=self.is_train=tf.placeholder(tf.bool,name='is_train')
            batch_size=self.args.batch_size
            n_gpu=len(self.args.gpu_list.split(','))

            x,m,y,self.pred_key = self.sample_queue.dequeue_many(batch_size)
            if self.args.is_train==False and batch_size==1:
                #predicting as server
                with tf.device('/gpu:0'):
                    _,self.prob = self._forward(x, is_train)
            else:
                with tf.device('/cpu:0'):
                    if self.args.is_train:
                        if self.args.start_lr is None:
                            opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
                        else:
                            lr = tf.train.exponential_decay(self.args.start_lr, self.global_step, 1500000, 0.5, staircase=True)
                            opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

                    batch_x=tf.split(value=x,num_or_size_splits=n_gpu,axis=0)
                    batch_m=tf.split(value=m,num_or_size_splits=n_gpu,axis=0)
                    batch_y=tf.split(value=y,num_or_size_splits=n_gpu,axis=0)

                tower_grads=[]
                tower_loss=[]
                tower_prob=[]
                for i in range(n_gpu):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                            logit,prob = self._forward([batch_x[i], batch_m[i]],is_train)
                            tower_prob.append(prob)
                            if self.args.is_train:
                                loss = self._loss(logit, batch_y[i])
                                grads = opt.compute_gradients(loss)
                                tower_grads.append(grads)
                                tower_loss.append(loss)
                self.prob=tf.reshape(tower_prob,[batch_size,2])
                if self.args.is_train:
                    grads = self._average_gradients(tower_grads)
                    apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
                    self.train_step = apply_gradient_op
                    self.loss=tower_loss[0]

                    #summary
                    if not self.args.start_lr is None:
                        tf.summary.scalar('lr',lr)
                    tf.summary.scalar('%s-classifier-loss'%self.name,self.loss)
                    correct_prediction = tf.equal(tf.argmax(self.prob, axis=1),y)
                    tf.summary.scalar('%s-classifier-acc'%self.name,tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
                    self.summary_step=tf.summary.merge_all()

            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

        

    def push_pipline(self,sentence,label,key):
        feed = {self.sentence:sentence,self.label:label,self.key:key}
        self.sess.run([self.feed_step], feed_dict=feed)

    def pop_pipline(self, is_train):
        pred,key = self.sess.run([self.pred,self.pred_key],feed_dict={self.is_train:False})
        return pred,key

    def offline_pred(self, sentences):
        if self.args.shuffle_input:
            print 'error, offline prediction can not shuffle'
            return

        for s in sentences:
            self.push_pipline(s,0,0)
        return self.pop_pipline()

