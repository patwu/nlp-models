all: seqtochar

TF_INC = /usr/local/lib/python2.7/dist-packages/tensorflow/include

seqtochar:
	g++ -I. -L. -std=c++11 --shared toids_op.cc -o toids_op.so -fPIC -I $(TF_INC) -O2

clean:
	rm -f *.o *.pyc

