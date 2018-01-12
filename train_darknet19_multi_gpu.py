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
    image_resized = tf.image.resize_images(image_decoded, [cfg.IMAGENET_IMAGE_SIZE, cfg.IMAGENET_IMAGE_SIZE])/255
    return image_resized, label


def get_train_path_set(split_ratio=0.9):
    class_name, class_index = cfg.pickup_dic()

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

    for i in range(len(image_filenames)):
        filename = os.path.split(image_filenames_shuffle[i])[1]
        headname, _ = os.path.splitext(filename)
        image_id = headname.split('_')[0]
        assert class_index[image_id] == image_label_shuffle[i]

    split = int(len(image_filenames_shuffle)*split_ratio)

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

    return info


def get_data(images_ph, label_ph, batch_size, num_parallel, num_split):

    dataset = tf.data.Dataset.from_tensor_slices((images_ph, label_ph))
    dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel)
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    imagenet_data, imagenet_label = iterator.get_next()
    imagenet_datas = tf.split(imagenet_data, num_split)
    imagenet_label = tf.split(imagenet_label, num_split)

    return imagenet_datas, imagenet_label, iterator


def get_loss(imagenet_data, imagenet_label, is_training):
    dark19_core, _ = yb.darknet19_core(imagenet_data, is_train=is_training)
    darknet19_output = yb.darnet19_body(dark19_core, is_train=is_training)
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=imagenet_label, logits=darknet19_output))
    loss_l2 = ce + tf.losses.get_regularization_loss()

    correct_prediction = tf.equal(tf.cast(tf.argmax(darknet19_output, 1), tf.int32), imagenet_label)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss_l2, ce, acc


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        epoch_start = 1
        learning_rate = 0.001
        batch_size = 6
        num_parallel = batch_size*2
        num_gpus = 1

        is_training = tf.placeholder(tf.bool, name='is_training')

        imagenet_dic = get_train_path_set()
        # cons_filenames = tf.constant(imagenet_dic["train_filename"])
        # cons_labels = tf.constant(imagenet_dic["train_label"])

        images_ph = tf.placeholder(tf.string, shape=[None], name='image_data')
        label_ph = tf.placeholder(tf.int32, shape=[None], name='imagenet_label')
        opt = tf.train.AdamOptimizer(learning_rate)

        imagenet_datas, imagenet_labels, iterator = get_data(images_ph, label_ph, batch_size, num_parallel, num_gpus)

        tower_grads = []
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                loss, ce, acc = get_loss(imagenet_datas[i], imagenet_labels[i], is_training)
                grads = opt.compute_gradients(loss)
                tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_darknet19_op = opt.apply_gradients(grads)

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        # tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)

        sess.run(tf.global_variables_initializer())

        if cfg.RESUME_DARKNET19_TRAIN == True:
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
        for epoch in range(epoch_start, 100):
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
                    loss_val, accuracy = sess.run([ce, acc], feed_dict={is_training: False})
                    loss_all += loss_val
                    acc_all += accuracy
                    count += 1
                    if count == int(imagenet_dic["test_size"] / batch_size):
                        break
                except tf.errors.OutOfRangeError:
                    break

            print(loss_all / count, acc_all / count)


main()
