import os.path
import numpy as np
import tensorflow as tf

class ToIdsTest(tf.test.TestCase):

    def test_seqtoids(self):
        library_filename = os.path.join(tf.resource_loader.get_data_files_path(),'toids_op.so')
        toids_op_module = tf.load_op_library(library_filename)
 
        with self.test_session():
            result=toids_op_module.toids("123456");
            print result[0].eval()
            print result[1].eval()

    def test_print_oplist(self):
        library_filename = os.path.join(tf.resource_loader.get_data_files_path(),'toids_op.so')
        toids_op_module = tf.load_op_library(library_filename)
        print toids_op_module.OP_LIST
 

if __name__ == '__main__':
    tf.test.main()
