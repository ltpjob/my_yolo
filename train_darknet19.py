import numpy as np
import os
import cv2
import tensorflow as tf
import trunk.config as cfg
import datetime
import time
import trunk.yolov2_body as yb
import trunk.yolov2_loss as yl
import random
import copy
import pickle

def minibatches_index(inputs=None, batch_size=None, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, 3)
    image_resized = tf.image.resize_images(image_decoded, [cfg.IMAGENET_IMAGE_SIZE, cfg.IMAGENET_IMAGE_SIZE])/255.0
    return image_resized, label


def class_stat(cls):

    cls_dic = {}
    for i in range(len(cls)):
        if cls[i] not in cls_dic:
            cls_dic[cls[i]] = 0
        else:
            cls_dic[cls[i]] += 1

    print('len dic:', len(cls_dic))
    print(cls_dic)


def check_filename_class(class_index, image_filenames, image_label):
    for i in range(len(image_filenames)):
        filename = os.path.split(image_filenames[i])[1]
        headname, _ = os.path.splitext(filename)
        image_id = headname.split('_')[0]
        # print(class_index[image_id], image_label[i])
        assert class_index[image_id] == image_label[i]


def get_train_path_set(split_ratio=0.9, reload=False):
    class_name, class_index = cfg.pickup_dic()

    if reload is False and os.path.exists(cfg.IMAGENET_DATA_DIC):
        pkl_file = open(cfg.IMAGENET_DATA_DIC, 'rb')
        info = pickle.load(pkl_file)
        pkl_file.close()

    else:
        image_filenames = []
        image_label = []
        for key, value in class_name.items():
            imagedir = os.path.join(cfg.IMAGENET_TRAINIMAGE, key)
            for parent, dirnames, filenames in os.walk(imagedir):
                for filename in filenames:
                    headname, expname = os.path.splitext(filename)
                    expname = expname.lower()
                    image_id = headname.split('_')[0]
                    full_path = os.path.join(imagedir, filename)

                    if expname == '.jpeg' and image_id == key:
                        image_filenames.append(full_path)
                        image_label.append(class_index[image_id])
                        print(full_path, len(image_filenames), len(image_label))

        roll_seed = random.randint(10, 100)
        random.seed(roll_seed)
        image_filenames_shuffle = copy.copy(image_filenames)
        random.shuffle(image_filenames_shuffle)
        random.seed(roll_seed)
        image_label_shuffle = copy.copy(image_label)
        random.shuffle(image_label_shuffle)

        check_filename_class(class_index, image_filenames_shuffle, image_label_shuffle)

        split = int(len(image_filenames_shuffle) * split_ratio)

        train_filename = image_filenames_shuffle[0:split]
        train_label = image_label_shuffle[0:split]
        test_filename = image_filenames_shuffle[split:]
        test_label = image_label_shuffle[split:]

        info = {
            "train_filename": train_filename,
            "train_label": train_label,
            "train_size": len(train_filename),
            "test_filename": test_filename,
            "test_label": test_label,
            "test_size": len(test_filename),
        }

        output = open(cfg.IMAGENET_DATA_DIC, 'wb')
        pickle.dump(info, output)
        output.close()

    check_filename_class(class_index, info["train_filename"], info["train_label"])
    check_filename_class(class_index, info["test_filename"], info["test_label"])
    class_stat(info["train_label"])
    class_stat(info["test_label"])

    return info


def main():

    epoch_start = 1
    learning_rate = 0.000001
    batch_size = 6
    num_parallel = batch_size*2

    is_training = tf.placeholder(tf.bool, name='is_training')

    imagenet_dic = get_train_path_set(reload=False)

    images_ph = tf.placeholder(tf.string, shape=[None], name='image_data')
    label_ph = tf.placeholder(tf.int32, shape=[None], name='imagenet_label')

    dataset = tf.data.Dataset.from_tensor_slices((images_ph, label_ph))
    dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    imagenet_data, imagenet_label = iterator.get_next()

    dark19_core, _ = yb.darknet19_core(imagenet_data, is_train=is_training)
    darknet19_output = yb.darnet19_body(dark19_core, is_train=is_training)

    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=imagenet_label, logits=darknet19_output))
    loss_l2 = ce + tf.losses.get_regularization_loss()

    correct_prediction = tf.equal(tf.cast(tf.argmax(darknet19_output, 1), tf.int32), imagenet_label)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_darknet19_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_l2)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    sess.run(tf.global_variables_initializer())

    if cfg.RESUME_DARKNET19_TRAIN is True:
        print("restore darknet19 model " + "!"*10)
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(cfg.DARKNET19_MODEL_SAVE_DIR)
        if model_file is not None:
            epoch_start = epoch_start+int(model_file.split('-')[1])
            print(model_file)
            saver.restore(sess, model_file)
            # print(sess.run(dark19_core, feed_dict={images_ph: np.ones((1, 224, 224, 3)), is_training: False}))

    print('train start!!!!! epoch_start:', epoch_start, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # Compute for 100 epochs.
    for epoch in range(epoch_start, 200):
        sess.run(iterator.initializer, feed_dict={images_ph: imagenet_dic["train_filename"],
                                                  label_ph: imagenet_dic["train_label"]})
        count = 0
        while True:
            try:
                sess.run(train_darknet19_op, feed_dict={is_training: True})
                count += 1
                # print(count)
                if count == int(imagenet_dic["train_size"] / batch_size):
                    count = 0
                    break
            except tf.errors.OutOfRangeError:
                break

        print("epoch:", epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(cfg.DARKNET19_MODEL_SAVE_DIR, cfg.DARKNET19_MODEL_FILE_NAME),
                   global_step=epoch)

        sess.run(iterator.initializer, feed_dict={images_ph: imagenet_dic["test_filename"],
                                                  label_ph: imagenet_dic["test_label"]})

        count = 0
        loss_all = 0
        acc_all = 0
        while True:
            try:
                loss, accuracy = sess.run([ce, acc], feed_dict={is_training: False})
                loss_all += loss
                acc_all += accuracy
                count += 1
                if count == int(imagenet_dic["test_size"] / batch_size):
                    break
            except tf.errors.OutOfRangeError:
                break

        print(loss_all / count, acc_all / count, count)


main()
