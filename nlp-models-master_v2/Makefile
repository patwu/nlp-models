all: seqtochar

TF_INC = /home/hadoop/.local/lib/python2.7/site-packages/tensorflow/include

seqtochar:
	g++ -D_GLIBCXX_USE_CXX11_ABI=0 -I. -L. -std=c++11 --shared toids_op.cc -o toids_op.so -fPIC -I $(TF_INC) -O2

clean:
	rm -f *.o *.pyc

