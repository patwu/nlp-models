import os
import sys
import time
import argparse
import importlib
import numpy as np
import threading
import grpc
import translate_pb2
import tensorflow as tf
import translate_utils

np.set_printoptions(threshold=np.inf)

class Server(translate_pb2.BetaTranslatorServicer):
    def __init__(self, args):
        # logging
        self.stat_lock = threading.Lock()
        self.qps = 0
        self.emv_time = 0.
        self.logging_cnt = 0
        self.st_time = time.time()
        self.qps_time = time.time()

        self.model_mutex=threading.Lock()
        module = importlib.import_module(args.type)

        model=module.seq2seq(args)
        model.build_model()
        model.load_model()
        self.model=model
        _,self.id2dest=translate_utils.load_vocabulary(args.dest_vocabulary)
        self.src2id,_=translate_utils.load_vocabulary(args.src_vocabulary)

    def translate(self, request, context):
        t1 = time.time()
        self.stat_lock.acquire()
        self.qps += 1
        self.stat_lock.release()

        if time.time() - self.qps_time > 2:
            self.stat_lock.acquire()
            if time.time() - self.qps_time > 2:
                params = (time.time() - self.st_time, self.qps / (time.time() - self.qps_time), self.emv_time * 1000)
                print "[%f] QPS=%f, EMV Service time= %fms" % params
                sys.stdout.flush()
                self.qps = 0
                self.qps_time = time.time()
            self.stat_lock.release()

        key = request.key
        ori_text=request.ori_text

        model=self.model
        text=translate_utils.vocabulary_encode(ori_text,self.src2id)
        
        ret=model.greedy_translate(text,len(text))
        ret_text=translate_utils.vocabulary_decode(ret,self.id2dest)
       
        reply=translate_pb2.TranslateReply()
        reply.key = key
        reply.tran_text=ret_text

        t5 = time.time()
        self.stat_lock.acquire()
        self.emv_time = self.emv_time * 0.999 + (t5 - t1) * 0.001
        self.stat_lock.release()
        return reply


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--type', type=str)
    argparser.add_argument('--model_path', type=str, default='')
    argparser.add_argument('--port', type=int, default=50051)
    argparser.add_argument('--gpu_list', type=str, default='0')
    argparser.add_argument('--src_vocabulary', type=str)
    argparser.add_argument('--dest_vocabulary', type=str)
    argparser.add_argument('--batch_size', type=int,default=1)
    argparser.add_argument('--is_train', type=bool,default=False)
    argparser.add_argument('--enc_max_len', type=int, default=16)
    argparser.add_argument('--dec_max_len', type=int, default=32)

    args = argparser.parse_args()
    print args
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list
    gpu_id_str='0'

    server = translate_pb2.beta_create_Translator_server(Server(args))
    server.add_insecure_port('[::]:' + str(args.port))
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)
    except KeyboardInterrupt:
        server.stop(0)
