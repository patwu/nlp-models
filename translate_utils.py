import argparse
import sys
import codecs
reload(sys)
# -*- coding: utf-8 -*-

sys.setdefaultencoding('utf-8')

def load_vocabulary(voc_file):
    char2id={'<UNK>':0,'<BOS>':1,'<EOS>':2}
    id2char=['<UNK>','<BOS>','<EOS>']
    with open(voc_file,'r') as f:
        buf=f.read()
        lines=buf.encode('utf-8').split('\n')
        for i,line in enumerate(lines):
            if len(line)==0:
                continue
            char2id[line.decode('utf-8')]=i+3
            id2char.append(line)
        f.close()
    return char2id,id2char

def vocabulary_encode(sentence,char2id):
    e=[1]
    sentence=list(sentence.decode('utf-8'))
    for c in sentence:
        if c in char2id:
            e.append(char2id[c])
        else:
            e.append(0)
    e.append(2)
    return e

def vocabulary_decode(arr,id2char):
    sentence=''
    for i in arr:
        sentence+=id2char[i]
    return sentence

def build_vocabulary(corpus_file, voc_file, column):
    voc={}
    with open(corpus_file,'rb') as f:
        buf = f.read()
        lines=buf.encode('utf-8')
        lines=lines.split('\n')
        for line in lines:
            if len(line)==0:
                break
            sentence=line.split('\t')[column]
            char=list(sentence.decode('utf-8'))
            for c in char:
                if not c in voc:
                    voc[c]=1
        f.close()
    with open(voc_file, 'w') as f:
        for c in voc:
            f.write('%s\n'%c.encode('utf-8'))

if __name__=='__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--corpus_file', type=str)
    argparser.add_argument('--vocabulary_file', type=str)
    argparser.add_argument('--column', type=int, default=0)
    args = argparser.parse_args()
    
    build_vocabulary(args.corpus_file, args.vocabulary_file, args.column)
