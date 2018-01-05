import os
import tensorflow as tf
import trunk.config as cfg
from trunk.yolov2_loss import *

def l2loss_test():
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)


    img = tf.placeholder(tf.float32, [1, 3, 3, 3], "img")
    data = np.ones([1, 3, 3, 3], np.float32)

    regulaer = tf.contrib.layers.l2_regularizer(scale=0.0005)


    X = tf.layers.conv2d(img, 1, (1, 1), kernel_regularizer=regulaer,
                             bias_regularizer=regulaer)

    loss = tf.reduce_mean(X)

    loss += tf.losses.get_regularization_loss()

    train_op = tf.train.AdamOptimizer().minimize(loss)

    sess.run(tf.global_variables_initializer())

    sess.run(train_op, feed_dict={img: data})


def main():
    # save_test()
    with tf.device('/gpu:0'):
        l2loss_test()



main()