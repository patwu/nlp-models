TF_INC = /usr/local/lib/python2.7/dist-packages/tensorflow/include

.PHONY: all tfop proto
all: tfop proto

tfop:
	g++ -I. -L. -std=c++11 --shared toids_op.cc -o toids_op.so -fPIC -I $(TF_INC) -O2

proto:
	$(MAKE) -C proto

clean:
	rm -f *.o *.pyc
	$(MAKE) -C proto clean
