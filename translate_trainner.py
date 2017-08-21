import sys
import numpy as np
import tensorflow as tf
import argparse
import os
import threading
import time
import importlib
import translate_utils

np.set_printoptions(threshold=np.inf)


def start_feed_from_file(file_name, model):
    src2id,id2src=translate_utils.load_vocabulary(args.src_vocabulary)
    dest2id,id2dest=translate_utils.load_vocabulary(args.dest_vocabulary)
 
    if args.epoch is None:
        epoch=1000
    else:
        epoch=args.epoch
    for _ in range(epoch):
        with open(file_name,'r') as input_file:
            buf=input_file.read()
            lines=buf.encode('utf-8')
            lines=buf.split('\n')[:-1]
            for i,line in enumerate(lines):
#            line=lines[0]
#            while True:
                (src,dest)=line.split('\t')
                src=translate_utils.vocabulary_encode(src,src2id)
                dest=translate_utils.vocabulary_encode(dest,dest2id)
                s_len=len(src)
                d_len=len(dest)
                if s_len>args.enc_max_len or d_len>args.dec_max_len:
                    continue
                s_mask=[1]*len(src)+[0]*(args.enc_max_len-s_len)
                d_mask=[1]*len(dest)+[0]*(args.dec_max_len-d_len)
                src=np.pad(src,(0,args.enc_max_len-s_len),mode='constant')
                dest=np.pad(dest,(0,args.dec_max_len-d_len),mode='constant')
                model.push_pipline(src,s_mask,s_len,dest,d_mask,d_len,0)
            input_file.close()
    model.sess.run(model.sample_queue.close())

def train():
    module = importlib.import_module(args.type)
    model = module.seq2seq(args)
    
    model.build_input()
    model.build_model()
    model.load_model() 

    if args.num_thread==1:
        feed_thread = threading.Thread(target=start_feed_from_file, args=[args.filename, model])
        feed_thread.daemon=True
        feed_thread.start()
    else:
        for i in range(args.num_thread):
            feed_thread = threading.Thread(target=start_feed_from_file, args=[args.filename+'.%02d'%i, model])
            feed_thread.daemon=True
            feed_thread.start()

    print 'start %s ...' %('training')
    sys.stdout.flush()
    
    summary_writer = tf.summary.FileWriter('%s.log'%args.type,model.sess.graph)
    try:
        while True:
            start_time=time.time()
            '''
            debug=model.sess.run(model.debug,feed_dict={model.is_train:True})
            print debug
            continue
            '''
            _, loss,global_step, summary= model.sess.run([model.train_step, model.loss, model.global_step, model.summary_step],feed_dict={model.is_train:True})
            if global_step % 100 == 0:
                summary_writer.add_summary(summary, global_step)
                duration = time.time() - start_time
                num_examples_per_step = args.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration
                format_str = ('loss %.3f %.1f examples/sec; %.3f sec/batch')
                print (format_str % (loss, examples_per_sec, sec_per_batch))
                sys.stdout.flush()
    
            if global_step % 50000 == 0 and global_step!=0:
                model.save_model()
    except tf.errors.OutOfRangeError:
        pass
    model.save_model()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--type', type=str)
    argparser.add_argument('--model_path', type=str, default=None)
    argparser.add_argument('--batch_size', type=int, default=4)
    argparser.add_argument('--gpu_list', type=str, default='0')
    argparser.add_argument('--filename', type=str, default=None)
    argparser.add_argument('--src_vocabulary', type=str)
    argparser.add_argument('--dest_vocabulary', type=str)
    argparser.add_argument('--shuffle_input', type=bool, default=True)
    argparser.add_argument('--start_lr', type=float, default=None)
    argparser.add_argument('--is_train', type=bool, default=True)
    argparser.add_argument('--grad_clip', type=float, default=None)
    argparser.add_argument('--epoch', type=int, default=None)
    argparser.add_argument('--num_thread', type=int ,default=1)
    argparser.add_argument('--enc_max_len', type=int,default=16)
    argparser.add_argument('--dec_max_len', type=int,default=32)


    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    train()

