rm -f toids_op.so

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

g++ -I. -L. -std=c++11 -shared toids_op.cc -o toids_op.so -fPIC -I $TF_INC -O2  

python toids_op_test.py
