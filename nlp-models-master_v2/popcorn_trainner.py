#_sum sys
import sys
import numpy as np
import tensorflow as tf
import argparse
import os
import threading
import time
import importlib
import re
import pinyin

def start_feed_from_file(file_name, model):
    bias=0
    if args.epoch is None:
        epoch=1000
    else:
        epoch=args.epoch
    for _ in range(epoch):
        with open(file_name,'r') as input_file:
            lines=input_file.readlines()
            for line in lines:
                line = line.lower()
                (_,label,sentence)=line.split('\t', 3)
                ### Chinese character to pinyin
                sentence = sentence.decode('utf8')
                zhpattern = re.compile(u'[\u4e00-\u9fa5]+')
                list1 = zhpattern.findall(sentence)
                if not list1:    
                    sentence = sentence.encode('utf8')
                else:
                    for word in list1:
                        sentence =re.sub(word, pinyin.get(word, format="numerical", delimiter=" "), sentence)
                    sentence = sentence.encode('utf8')
                ###     ###
                sentence=sentence.strip()
                model.push_pipline(sentence,label,0)
            input_file.close()
    model.sess.run(model.sample_queue.close())


def train():
    ### training model
    module = importlib.import_module(args.type)
    model = module.Classifier(args)

    library_filename = os.path.join('./','toids_op.so') #tf.resource_loader.get_data_files_path()
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


    print ('start %s ...' %('training'))
    sys.stdout.flush()
    
    summary_writer = tf.summary.FileWriter('%s.log'%args.type,model.sess.graph)

    ### test model
    if args.test_filename is not None:
	module1 = importlib.import_module(args.type)
	model1 = module1.Classifier(args)
	model1.build_input(op_module.toids)
	model1.build_model()
	    
	if args.num_thread==1:
	    feed_thread1 = threading.Thread(target=start_feed_from_file, args=[args.test_filename, model1])
	    feed_thread1.daemon=True
	    feed_thread1.start()
	else:
	    for i in range(args.num_thread):
	        feed_thread1 = threading.Thread(target=start_feed_from_file, args=[args.test_filename+'.%02d'%i, model1])
		feed_thread1.daemon=True
		feed_thread1.start()
    
    
    try:
        while True:
            start_time=time.time()
            _, loss,global_step, _accuracy, summary= model.sess.run([model.train_step, model.loss, 
                model.global_step, model.accuracy, model.summary_step],feed_dict={model.is_train: True})
            #print debug
            if global_step % 10 == 0:
                #summary_writer.add_summary(summary, global_step)
    
                duration = time.time() - start_time
                num_examples_per_step = args.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration
                format_str = ('global_step %.1f; loss %.3f %.1f examples/sec; %.3f sec/batch; accuracy %.3f')
                print (format_str % (global_step, loss, examples_per_sec, sec_per_batch, _accuracy))
                
                sys.stdout.flush()
    
            if global_step % 50 == 0 and global_step!=0:
                model.save_model()
            
            if args.test_filename is not None:
		if global_step % 50 == 0:    
	            model1.load_model() 
	            sys.stdout.flush()

	            loss, _accuracy= model1.sess.run([model1.loss, model1.accuracy],feed_dict={model1.is_train: False})
		    format_str = ('TEST_EVALUATE: loss %.3f; accuracy %.3f')
		    print ("\n" + format_str % (loss, _accuracy) + "\n")
                        
    except tf.errors.OutOfRangeError:
        pass
    model.save_model()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--type', type=str)
    argparser.add_argument('--model_path', type=str, default=None)
    argparser.add_argument('--batch_size', type=int, default=128) #32
    argparser.add_argument('--gpu_list', type=str, default='0')
    argparser.add_argument('--filename', type=str, default=None)
    argparser.add_argument('--test_filename', type=str, default=None)
    argparser.add_argument('--shuffle_input', type=bool, default=True) #True
    argparser.add_argument('--start_lr', type=float, default=None)
    argparser.add_argument('--is_train', type=bool, default=True)
    argparser.add_argument('--grad_clip', type=float, default=None)
    argparser.add_argument('--epoch', type=int, default=None)
    argparser.add_argument('--num_thread', type=int ,default=1)

    args = argparser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    
    train()
