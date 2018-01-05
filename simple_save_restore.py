import os
import tensorflow as tf
import trunk.config as cfg
from trunk.yolov2_loss import *

sess = tf.InteractiveSession()

def DarknetConv2D_BN_Leaky(X, filters, kernel_size, strides=(1, 1), is_train=True, padding='same'):

    X = tf.layers.conv2d(X, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    X = tf.layers.batch_normalization(X, training=is_train)
    X = tf.maximum(0.1 * X, X)

    return X



def network1(X, reuse=None, is_train=True):
    with tf.variable_scope("model", reuse=reuse):
        conv1 = DarknetConv2D_BN_Leaky(X, 1, (1, 1), (1, 1), is_train)
        conv2 = DarknetConv2D_BN_Leaky(conv1, 1, (1, 1), (1, 1), is_train)

    return conv2

def network2(X, reuse=None, is_train=True):
    with tf.variable_scope("model", reuse=reuse):
        conv1 = DarknetConv2D_BN_Leaky(X, 1, (1, 1), (1, 1), is_train)
        conv2 = DarknetConv2D_BN_Leaky(conv1, 1, (1, 1), (1, 1), is_train)
        conv3 = DarknetConv2D_BN_Leaky(conv2, 1, (1, 1), (1, 1), is_train)
        conv4 = DarknetConv2D_BN_Leaky(conv3, 1, (1, 1), (1, 1), is_train)

    return conv2, conv4


def save_test():
    is_training = tf.placeholder(tf.bool)
    img = tf.placeholder(tf.float32, [1, 3, 3, 3], "img")
    data = np.ones([1, 3, 3, 3], np.float32)

    net = network1(img, is_train=is_training)

    sess.run(tf.global_variables_initializer())

    res = sess.run(net, feed_dict={img:data, is_training:0})

    print("test:", res)

    saver = tf.train.Saver()
    save_path = saver.save(sess, os.path.join(cfg.TEST_MODEL_SAVE_DIR, cfg.MODEL_FILE_NAME), global_step=99)

def restore_test():
    is_training = tf.placeholder(tf.bool)
    img = tf.placeholder(tf.float32, [1, 3, 3, 3], "img")
    data = np.ones([1, 3, 3, 3], np.float32)

    net = network1(img, is_train=is_training)
    net1, net2 = network2(img, reuse=tf.AUTO_REUSE, is_train=is_training)
    train = tf.reduce_mean(net1)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(0.01).minimize(train)

    sess.run(tf.global_variables_initializer())

    model_file = tf.train.latest_checkpoint(cfg.TEST_MODEL_SAVE_DIR)

    ckpt_vars = [t[0] for t in tf.contrib.framework.list_variables(model_file)]
    print(ckpt_vars)
    vars_to_restore = []
    for v in tf.global_variables():
        if v.name[:-2] in ckpt_vars:
            vars_to_restore.append(v)

    saver = tf.train.Saver(vars_to_restore)
    saver.restore(sess, model_file)

    res = sess.run(net1, feed_dict={img: data, is_training: 0})
    res1 = sess.run(net2, feed_dict={img: data, is_training: 0})
    print("restore:", res)
    print("newnetwork:", res1)

    sess.run(train_op, feed_dict={img: data, is_training: 1})
    sess.run(train_op, feed_dict={img: data, is_training: 1})
    sess.run(train_op, feed_dict={img: data, is_training: 1})
    sess.run(train_op, feed_dict={img: data, is_training: 1})


    after_train = sess.run(net, feed_dict={img: data, is_training: 0})
    after_train1 = sess.run(net1, feed_dict={img: data, is_training: 0})

    print("after_train:", after_train)
    print("after_train1", after_train1)

def main():
    # save_test()
    restore_test()



main()