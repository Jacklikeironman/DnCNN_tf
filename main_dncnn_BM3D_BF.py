from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model_dncnn_BM3D_BF import DNCNN
import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 4, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 40, "The size of image to use [40]")
flags.DEFINE_integer("label_size", 40, "The size of label to produce [40]")
flags.DEFINE_float("lr_init", 1e-3, "The learning rate of gradient descent algorithm [1e-3]")
flags.DEFINE_integer("depth", 17, "Depth of Network. 17]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
'''
如果需要输入为BM3D去噪后的图片，
请将   'train_images_dir'设为"data/Train400-25"
       'test_images_dir'设为'data/Test/Set12_BM3D_25'
       "checkpoint_dir"设为"checkpoint_dncnn_BM3D"
       "result_dir"设为"result_dncnn_BM3D"
       "log_dir"设为'Log_dncnn_BM3D'

如果需要输入为双边滤波去噪后的图片，
请将   'train_images_dir'设为"data/BF-Train400-25"
       'test_images_dir'设为'data/Test/BF-Set12-25'
       "checkpoint_dir"设为"checkpoint_dncnn_BF"
       "result_dir"设为"result_dncnn_BF"
       "log_dir"设为'Log_dncnn_BF'
       
注意model脚本中save和load函数中的名字，以及test函数中的输出字符都要进行相应的修改
'''
flags.DEFINE_string("train_images_dir", "data/Train400-25", "Name of train images directory [data/Train400-25]")
flags.DEFINE_string("train_labels_dir", "data/Train400", "Name of train labels directory [data/Train400]")
flags.DEFINE_string("test_images_dir", "data/Test/Set12_BM3D_25", "Name of test images directory [data/Test/Set12_BM3D_25]")
flags.DEFINE_string("test_labels_dir", "data/Test/Set12", "Name of test labels directory [data/Test/Set12]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_dncnn_BM3D", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("result_dir", "result_dncnn_BM3D", "Name of result directory [result]")
flags.DEFINE_string("log_dir", 'Log_dncnn_BM3D', "Name of log directory[Log]")
flags.DEFINE_boolean("is_train", True, "BN parameters and processing mode,True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    with tf.Session() as sess:
        dncnn = DNCNN(sess,FLAGS)
        if FLAGS.is_train:
            dncnn.train()
        else:
            dncnn.test()

if __name__ == '__main__':
    tf.app.run()