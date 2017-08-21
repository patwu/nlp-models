import sys
import translate_pb2
import argparse
import threading
import codecs

from grpc.beta import implementations

from threadutils import Worker 
from threadutils import Counter
from threadutils import ThreadPool

def work(src,idx,counter):
    stub = translate_pb2.beta_create_Translator_stub(channel) 
    req=translate_pb2.TranslateRequest()
    req.ori_text=src
    req.key=idx
    result=stub.translate(req,5000)
    if args.verbose==1:
        print src, result.tran_text

    #TODO BLEU calc

def test():
    total=args.num_sample
    labels=[0]*total
    result=[0]*total
    n_sample=0
    pool=ThreadPool(args.num_thread)
    counter=Counter()

    cnt=args.num_skip
    with open(args.test_file,'r') as input_file:
        buf=input_file.read()
        #lines=buf.encode('utf-8')
        lines=buf.split('\n')[:-1]
        for line in lines[args.num_skip:]:
            src,dest=line.split('\t')
            result[n_sample]=dest
            pool.add_task(work,src,n_sample,counter)
            n_sample+=1
            if n_sample%100==0:
                bleu,_=counter.get()
                print '%d %.3f'%(n_sample,(0.+bleu)/n_sample)
            if n_sample==total:
                break
        pool.wait_completion() 
        bleu,_=counter.get()
        print '%d %.3f'%(n_sample,(0.+bleu)/n_sample)

if __name__=='__main__':
    
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--port', type=int, default=30031)
    argparser.add_argument('--host', type=str, default='localhost')
    argparser.add_argument('--num_thread', type=int, default=1)
    argparser.add_argument('--test_file', type=str)
    argparser.add_argument('--num_sample', type=int, default=100)
    argparser.add_argument('--verbose', type=int, default=1)
    argparser.add_argument('--num_skip', type=int,default=0)
    args = argparser.parse_args()

    global channel
    global results

    channel = implementations.insecure_channel(args.host,args.port)
    test()

