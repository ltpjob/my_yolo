import numpy as np
import os
import cv2
import tensorflow as tf
import trunk.config as cfg
import datetime
import time
import trunk.yolov2_body as yb
import trunk.yolov2_loss as yl


def image_label_show(image, scores, boxes, classes, name='image', delay=500):

    image = image.copy()

    for i in range(len(boxes)):
        cv2.rectangle(image, (boxes[i][1], boxes[i][0]),
                      (boxes[i][3], boxes[i][2]),
                      (0, 0, 255), 2)

    cv2.imshow(name, image)
    cv2.waitKey(delay)


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


def restore_darknet19_core(sess):
    model_file = tf.train.latest_checkpoint(cfg.DARKNET19_MODEL_SAVE_DIR)

    ckpt_vars = [t[0] for t in tf.contrib.framework.list_variables(model_file)]
    vars_to_restore = []
    for v in tf.global_variables():
        if v.name[:-2] in ckpt_vars:
            vars_to_restore.append(v)

    saver = tf.train.Saver(vars_to_restore)
    saver.restore(sess, model_file)


def image_trueboxs_to_corners(trueboxs):

    boxes = np.zeros(trueboxs.shape)
    boxes[:, 0] = trueboxs[:, 0] - trueboxs[:, 2] / 2
    boxes[:, 1] = trueboxs[:, 1] + trueboxs[:, 3] / 2
    boxes[:, 2] = trueboxs[:, 0] + trueboxs[:, 2] / 2
    boxes[:, 3] = trueboxs[:, 1] - trueboxs[:, 3] / 2

    # print(boxes)

    boxes = (boxes*cfg.IMAGE_SIZE).astype(int)

    return boxes


def image_truebox_show(image, trueboxs, delay=500, name='image'):

    image = image.copy()

    boxes = image_trueboxs_to_corners(trueboxs)
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 0, 255), 2)

    cv2.imshow(name, image)
    cv2.waitKey(delay)


def main():
    epoch_start = 1
    learning_rate = 0.0001

    images_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image_data')
    matching_true_boxes = tf.placeholder(
        tf.float32, shape=[None, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT],
        name='matching_true_boxes')
    true_boxes = tf.placeholder(
        tf.float32, shape=[None, cfg.MAX_TRUEBOXS, 4], name='true_boxes')
    is_training = tf.placeholder(tf.bool, name='is_training')

    dark19_core, path_1 = yb.darknet19_core(images_ph, is_train=is_training)
    yolov2_output = yb.yolov2_body(dark19_core, path_1, is_train=is_training)
    scores, boxes, classes = yb.predic(yolov2_output)
    total_loss, confidence_loss, classification_loss, coordinates_loss = \
        yl.yolo_loss(yolov2_output, matching_true_boxes, true_boxes)

    total_loss_l2 = total_loss + tf.losses.get_regularization_loss()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate, name="yolov2_op").minimize(total_loss_l2)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    sess.run(tf.global_variables_initializer())

    if cfg.RESUME_DARKNET19_CORE == True:
        print("restore darknet19_core!!!!")
        restore_darknet19_core(sess)
        # print(sess.run(dark19_core, feed_dict={images_ph: np.ones((1, 224, 224, 3)), is_training: False}))

    if cfg.RESUME_YOLOV2_TRAIN == True and cfg.RESUME_DARKNET19_CORE != True:
        print("restore yolov2 model " + "!"*10)
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(cfg.YOLOV2_MODEL_SAVE_DIR)
        if model_file is not None:
            epoch_start = epoch_start+int(model_file.split('-')[1])
            print(model_file)
            saver.restore(sess, model_file)

    tf.summary.scalar('total_loss_l2', total_loss_l2)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('confidence_loss', confidence_loss)
    tf.summary.scalar('classification_loss', classification_loss)
    tf.summary.scalar('coordinates_loss', coordinates_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    merged_tb = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(cfg.LOGS_TRAIN, sess.graph)
    test_writer = tf.summary.FileWriter(cfg.LOGS_TEST, sess.graph)

    image_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.IMAGE_DATA))
    # np.random.seed(0)
    # np.random.shuffle(image_data)
    label_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.LABEL_DATA))
    # np.random.seed(0)
    # np.random.shuffle(label_data)
    trueboxs_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.TRUEBOX_DATA))
    # np.random.seed(0)
    # np.random.shuffle(trueboxs_data)

    # for i in range(1000):
    #     idx = np.arange(image_data.shape[0])
    #     np.random.shuffle(idx)
    #     print(idx[i])
    #     image_truebox_show(image_data[idx[i]], trueboxs_data[idx[i]], delay=0)

    split = int(image_data.shape[0] * 0.9)
    print(split, image_data.shape[0])

    train_data = image_data[:split]
    train_label = label_data[:split]
    train_trueboxs = trueboxs_data[:split]

    test_data = image_data[split:]
    test_label = label_data[split:]
    test_trueboxs = trueboxs_data[split:]

    data_size = 30
    start = 0
    end = start+data_size

    # for i in range(start, end):
    #     scrs, bxes, clas = sess.run([scores, boxes, classes],
    #                                 feed_dict={images_ph: np.expand_dims(test_data[i], axis=0) / 255.0,
    #                                            is_training: False})
    #     print(scrs, bxes, clas)
    #     image_label_show(test_data[i], scrs, bxes, clas, name='train' + str(i), delay=0)
    #
    # cv2.waitKey()

    print('train start!!!!!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(epoch_start, 2000):
        idxs = np.arange(train_data.shape[0])

        oldtime = datetime.datetime.now()
        for idx in minibatches_index(idxs, 6, shuffle=True):
            sess.run([train_op], feed_dict={images_ph: train_data[idx]/255.0,
                                            true_boxes: train_trueboxs[idx],
                                            matching_true_boxes: train_label[idx],
                                            is_training: True})

        newtime = datetime.datetime.now()
        print("epoch:", epoch, "time_cost:", (newtime - oldtime).seconds,
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        if epoch % 1 == 0 and epoch != 0:
            saver = tf.train.Saver()
            saver.save(sess, save_path=os.path.join(cfg.YOLOV2_MODEL_SAVE_DIR, cfg.YOLOV2_MODEL_FILE_NAME),
                       global_step=epoch)

        train_tb, total_loss_train, train_confidence_loss, train_classification_loss, train_coordinates_loss, result_train = \
            sess.run([merged_tb, total_loss, confidence_loss, classification_loss, coordinates_loss, yolov2_output],
                     feed_dict={images_ph: train_data[start:end] / 255.0,
                                true_boxes: train_trueboxs[start:end],
                                matching_true_boxes: train_label[start:end],
                                is_training: False})

        print("epoch:", epoch, "train_total_loss:", total_loss_train,
              "train_conf_loss:", train_confidence_loss,
              "train_class_loss:", train_classification_loss,
              "train_coor_loss:", train_coordinates_loss)

        test_tb, test_total_loss, test_confidence_loss, test_classification_loss, test_coordinates_loss, result_test = \
            sess.run([merged_tb, total_loss, confidence_loss, classification_loss, coordinates_loss, yolov2_output],
                     feed_dict={images_ph: test_data[start:end]/255.0,
                                true_boxes: test_trueboxs[start:end],
                                matching_true_boxes: test_label[start:end],
                                is_training: False})
        print("epoch:", epoch, "test_total_loss :", test_total_loss,
              "test_conf_loss :", test_confidence_loss,
              "tset_class_loss :", test_classification_loss,
              "test_coor_loss :", test_coordinates_loss)

        train_writer.add_summary(train_tb, epoch)
        test_writer.add_summary(test_tb, epoch)

        scrs, bxes, clas = sess.run([scores, boxes, classes],
                                    feed_dict={images_ph: np.expand_dims(train_data[start+5], axis=0) / 255.0, is_training: False})
        print(scrs, bxes, clas)
        image_label_show(train_data[start + 5], scrs, bxes, clas, name='train')

        scrs, bxes, clas = sess.run([scores, boxes, classes],
                                    feed_dict={images_ph: np.expand_dims(test_data[start+5], axis=0) / 255.0, is_training: False})
        print(scrs, bxes, clas)
        image_label_show(test_data[start + 5], scrs, bxes, clas, name='test')


main()






