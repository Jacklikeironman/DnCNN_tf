#! /usr/bin/python
# -*- coding: utf8 -*-

from data_generator_original import datagenerator,imsave
import time
import os
import glob
import cv2
import tensorflow as tf
import numpy as np

class DNCNN(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.label_size = config.label_size
        self.c_dim = config.c_dim
        self.lr_init = config.lr_init
        self.epoch = config.epoch

        self.train_labels_dir = config.train_labels_dir
        self.test_labels_dir = config.test_labels_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.result_dir = config.result_dir
        self.log_dir = config.log_dir
        self.is_train = config.is_train
        self.depth = config.depth

        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')

        self.pred = self.network()
        self.loss = tf.losses.mean_squared_error(self.labels, self.pred)
        self.metric = tf.multiply(10.0, tf.log(1.0 * 1.0 / self.loss) / tf.log(10.0))

        self.saver = tf.train.Saver()

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('PSNR', self.metric)
        self.merged_summary_op = tf.summary.merge_all()

    def WeightsVariable(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=1e-3), name='weights')

    def BiasVariable(self, shape):
        return tf.Variable(tf.zeros(shape), name='biases')

    def network(self):
        layer_count = 1
        with tf.name_scope('conv' + str(layer_count)):
            weights = self.WeightsVariable([3, 3, self.c_dim, 64])
            biases = self.BiasVariable([64])
            conv = tf.nn.conv2d(self.images, weights, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.relu(tf.nn.bias_add(conv, biases))
        for i in range(self.depth - 2):
            layer_count += 1
            with tf.name_scope('conv' + str(layer_count)):
                weights = self.WeightsVariable([3, 3, 64, 64])
                conv = tf.nn.conv2d(conv, weights, strides=[1, 1, 1, 1], padding='SAME')
                conv = tf.layers.batch_normalization(conv, axis=3, momentum=0.0, epsilon=0.0001, training=self.is_train)
                conv = tf.nn.relu(conv)
        layer_count += 1
        with tf.name_scope('conv' + str(layer_count)):
            weights = self.WeightsVariable([3, 3, 64, self.c_dim])
            conv = tf.nn.conv2d(conv, weights, strides=[1, 1, 1, 1], padding='SAME')
        output = self.images + conv
        #output = self.images - conv
        return output

    def train(self):
        train_labels = datagenerator(GT_data_dir=self.train_labels_dir)
        train_labels = train_labels / 255.0

        self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(self.lr_init).minimize(self.loss)

        tf.initialize_all_variables().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print("Training...")

        for ep in range(self.epoch):
            if ep % 5 ==0 and ep != 0:
                self.lr_init = self.lr_init / 10
            batch_idxs = len(train_labels) // self.batch_size
            for idx in range(0, batch_idxs):
                batch_labels = train_labels[idx * self.batch_size: (idx + 1) * self.batch_size]
                noise = np.random.normal(0, 25 / 255.0, batch_labels.shape)
                batch_images = batch_labels + noise

                counter += 1

                _, err, psnr = self.sess.run([self.train_op, self.loss, self.metric],
                                             feed_dict={self.images:batch_images, self.labels:batch_labels})

                summary = self.sess.run(self.merged_summary_op,
                                        feed_dict={self.images:batch_images, self.labels:batch_labels})
                self.summary_writer.add_summary(summary, counter)

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f], PSNR: [%.4f], lr: [%.6f]" % (
                        (ep + 1), counter, time.time() - start_time, err, psnr, self.lr_init))
                if counter % 500 == 0:
                    self.save(self.checkpoint_dir, counter)

    def test(self):
        GT_file_list = sorted(glob.glob(self.test_labels_dir + '/*.png'))

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print("Testing...")

        for i in range(len(GT_file_list)):
            GT_img = cv2.imread(GT_file_list[i], 0) / 255
            GT_img =GT_img.reshape(1, GT_img.shape[0], GT_img.shape[1], self.c_dim)
            noise = np.random.normal(0, 25 / 255.0, GT_img.shape)
            test_img = GT_img + noise

            result = self.pred.eval({self.images:test_img, self.labels:GT_img})

            # MSE_DNCNN = np.float32(np.mean(np.square(result - GT_img)))
            MSE_DNCNN = np.float32(np.mean(np.square(np.uint8(result*255)/255 - np.uint8(GT_img*255)/255)))
            PSNR_DNCNN = np.multiply(10.0, np.log(1.0 * 1.0 / MSE_DNCNN) / np.log(10.0))
            print('Picture[%d]   MSE:[%.8f]   PSNR:[%.4f] ---------DNCNN' % ((i + 1), MSE_DNCNN, PSNR_DNCNN))

            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), self.result_dir)
            image_path = os.path.join(image_path, "test_image" + str(i) + ".png")
            # print(result)
            imsave(result, image_path)
            # print('*' * 40)
            # print(cv2.imread(image_path) / 255)
            test_img = test_img.squeeze()
            image_path2 = os.path.join(os.getcwd(), self.result_dir)
            image_path2 = os.path.join(image_path2, "noised_image" + str(i) + ".png")
            imsave(test_img, image_path2)




    def save(self, checkpoint_dir, step):
        model_name = "DNCNN.model"
        model_dir = "%s_%s" % ("dncnn", "sigma25")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("dncnn", "sigma25")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)


        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print('ckptpath: %s' % ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False