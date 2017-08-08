rm -f toids_op.so

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC
#g++ -I. -L. -std=c++11 -shared toids_op.cc -o toids_op.so -fPIC -I $TF_INC -O2
g++ -I. -L. -std=c++11 -shared toids_op.cc -o toids_op.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 
python toids_op_test.py
