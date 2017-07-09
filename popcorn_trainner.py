_sum sys
import numpy as np
import tensorflow as tf
import argparse
import os
import threading
import time
import importlib

def start_feed_from_file(file_name, model):
    bias=0
    if args.epoch is None:
        epoch=1000
    else:
        epoch=args.epoch
    for _ in range(epoch):
        with open(file_name,'r') as input_file:
            lines=input_file.readlines()
            line=lines[1]
            for line in lines:
                (_,label,sentence)=line.split('\t')
                sentence=sentence.strip()
                model.push_pipline(sentence,label,0)
            input_file.close()
    model.sess.run(model.sample_queue.close())


def train():
    module = importlib.import_module(args.type)
    model = module.Classifier(args)
    
    library_filename = os.path.join(tf.resource_loader.get_data_files_path(),'toids_op.so')
    op_module = tf.load_op_library(library_filename)

    model.build_input(op_module.toids)
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
            _, loss,global_step, summary = model.sess.run([model.train_step, model.loss, model.global_step, model.summary_step],feed_dict={model.is_train:True})
            print debug
            if global_step % 10 == 0:
                summary_writer.add_summary(summary, global_step)
    
                duration = time.time() - start_time
                num_examples_per_step = args.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration
                format_str = ('loss %.3f %.1f examples/sec; %.3f sec/batch')
                print (format_str % (loss, examples_per_sec, sec_per_batch))
                sys.stdout.flush()
    
            if global_step % 10000 == 0 and global_step!=0:
                model.save_model()
    except tf.errors.OutOfRangeError:
        pass
    model.save_model()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--type', type=str)
    argparser.add_argument('--model_path', type=str, default=None)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--gpu_list', type=str, default='0')
    argparser.add_argument('--filename', type=str, default=None)
    argparser.add_argument('--shuffle_input', type=bool, default=True)
    argparser.add_argument('--start_lr', type=float, default=None)
    argparser.add_argument('--is_train', type=bool, default=True)
    argparser.add_argument('--grad_clip', type=float, default=None)
    argparser.add_argument('--epoch', type=int, default=None)
    argparser.add_argument('--num_thread', type=int ,default=1)

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    train()

