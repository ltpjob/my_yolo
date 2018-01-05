import numpy as np
import os
import tensorflow as tf
import cv2
import tensorflow as tf
import tensorlayer as tl
import trunk.config as cfg
from tensorlayer.layers import *
import datetime
import time
from tensorlayer.ops import open_tb


sess = tf.InteractiveSession()
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    box_scores = box_confidence * box_class_probs
    ### END CODE HERE ###

    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    box_classes = tf.argmax(box_scores, axis=-1)
    box_class_scores = tf.reduce_max(box_scores, axis=-1)
    ### END CODE HERE ###

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ### START CODE HERE ### (≈ 1 line)
    filtering_mask = box_class_scores >= threshold
    ### END CODE HERE ###

    # Step 4: Apply the mask to scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    ### END CODE HERE ###

    return scores, boxes, classes


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    # max_boxes_tensor = tf.Variable(max_boxes, dtype=tf.int32)  # tensor to be used in tf.image.non_max_suppression()
    # tf.InteractiveSession().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor
    max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    ### START CODE HERE ### (≈ 1 line)
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)
    ### END CODE HERE ###

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    ### END CODE HERE ###

    return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.concat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ], axis=-1)


def DarknetConv2D_BN_Leaky(X, filters, kernel_size, strides=(1, 1), is_train=True, padding='SAME', name="DBR"):

    X = Conv2d(X, filters, kernel_size, strides, padding=padding, name='conv2d_'+name)
    X = BatchNormLayer(X, is_train=is_train, act=lambda x: tl.act.lrelu(x, 0.1), name='bn_relu_'+name)

    return X


def yolo_head(feats):
    """Convert final layer features to bounding box parameters.

    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.

    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """

    conv_index = np.zeros((cfg.CELL_SIZE, cfg.CELL_SIZE, 2))
    for i in range(cfg.CELL_SIZE):
        for j in range(cfg.CELL_SIZE):
            conv_index[i, j] = [i, j]
    conv_index = np.expand_dims(conv_index, 2)
    conv_index = tf.constant(conv_index, tf.float32)

    anchors_tensor = np.array(cfg.ANCHOR_BOXES)
    anchors_tensor = tf.constant(anchors_tensor, tf.float32)

    conv_dims = np.array([cfg.CELL_SIZE, cfg.CELL_SIZE])

    box_confidence = tf.sigmoid(feats[..., 0:1])
    box_xy = tf.sigmoid(feats[..., 1:3])
    box_wh = tf.exp(feats[..., 3:5])
    box_class_probs = tf.nn.softmax(feats[..., 5:])

    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_loss(yolo_output, matching_true_boxes, true_boxes, rescore_confidence=False):
    # with tf.device('/cpu:0'):
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1

    obj_mask = matching_true_boxes[..., 0:1]
    noobj_mask = 1-obj_mask
    detectors_mask = obj_mask

    pred_xy, pred_wh, pred_confidence, pred_class_prob = yolo_head(yolo_output)

    pred_boxes = tf.concat(
        (tf.sigmoid(yolo_output[..., 1:3]), yolo_output[..., 3:5]), axis=-1)

    pred_xy = tf.expand_dims(pred_xy, 4)
    pred_wh = tf.expand_dims(pred_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    true_boxes_shape = tf.shape(true_boxes)

    # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
    true_boxes = tf.reshape(true_boxes, [
        true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
    ])
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    # Find IOU of each predicted box with each ground truth box.
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    # Best IOUs for each location.
    best_ious = tf.reduce_max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = tf.expand_dims(best_ious, axis=-1)

    # A detector has found an object if IOU > thresh for some true box.
    object_detections = tf.cast(best_ious > 0.6, tf.float32)


    # TODO: Darknet region training includes extra coordinate loss for early
    # training steps to encourage predictions to match anchor priors.

    # Determine confidence weights from object and no_object weights.
    # NOTE: YOLO does not use binary cross-entropy here.
    no_object_weights = (no_object_scale * (1 - object_detections) *
                         (1 - detectors_mask))
    no_objects_loss = no_object_weights * tf.square(-pred_confidence)

    if rescore_confidence:
        objects_loss = (object_scale * detectors_mask *
                        tf.square(best_ious - pred_confidence))
    else:
        objects_loss = (object_scale * detectors_mask *
                        tf.square(1 - pred_confidence))
    confidence_loss = objects_loss + no_objects_loss

    # Classification loss for matching detections.
    # NOTE: YOLO does not use categorical cross-entropy loss here.
    matching_classes = matching_true_boxes[..., 5:]
    classification_loss = (class_scale * detectors_mask *
                           tf.square(matching_classes - pred_class_prob))

    matching_boxes = matching_true_boxes[..., 1:5]
    coordinates_loss = (coordinates_scale * detectors_mask *
                        tf.square(matching_boxes - pred_boxes))

    confidence_loss_mean = tf.reduce_mean(tf.reduce_sum(confidence_loss, axis=[1, 2, 3, 4]))
    classification_loss_mean = tf.reduce_mean(tf.reduce_sum(classification_loss, axis=[1, 2, 3, 4]))
    coordinates_loss_mean = tf.reduce_mean(tf.reduce_sum(coordinates_loss, axis=[1, 2, 3, 4]))
    total_loss = confidence_loss_mean + classification_loss_mean + coordinates_loss_mean

    L2_regular = 0
    for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=True):  # [-3:]:
        if w.name.find("conv2d_darknet_body") == -1:
            print("yolo network L2:", w.name)
            L2_regular += tf.contrib.layers.l2_regularizer(5e-4)(w)

    total_loss += L2_regular

    return total_loss, confidence_loss_mean, classification_loss_mean, coordinates_loss_mean


def yolo_build_model(x, reuse, is_train=True):

    with tf.variable_scope("model", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        tl.ops.disable_print()

        X_input = tl.layers.InputLayer(x, name='input')
        #416to208
        X = DarknetConv2D_BN_Leaky(X_input, 32, (3, 3), strides=(1, 1), is_train=is_train, name='1')
        X = MaxPool2d(X, (2, 2), padding='VALID', name='pool_1')

        #208to104
        X = DarknetConv2D_BN_Leaky(X, 64, (3, 3), strides=(1, 1), is_train=is_train, name='2')
        X = MaxPool2d(X, (2, 2), padding='VALID', name='pool_2')

        # 104to52
        X = DarknetConv2D_BN_Leaky(X, 128, (3, 3), strides=(1, 1), is_train=is_train, name='3')
        X = DarknetConv2D_BN_Leaky(X, 64, (1, 1), strides=(1, 1), is_train=is_train, name='4')
        X = DarknetConv2D_BN_Leaky(X, 128, (3, 3), strides=(1, 1), is_train=is_train, name='5')
        X = MaxPool2d(X, (2, 2), padding='VALID', name='pool_3')

        #52to26
        X = DarknetConv2D_BN_Leaky(X, 256, (3, 3), strides=(1, 1), is_train=is_train, name='6')
        X = DarknetConv2D_BN_Leaky(X, 128, (1, 1), strides=(1, 1), is_train=is_train, name='7')
        X = DarknetConv2D_BN_Leaky(X, 256, (3, 3), strides=(1, 1), is_train=is_train, name='8')
        X = MaxPool2d(X, (2, 2), padding='VALID', name='pool_4')

        #26to13
        X = DarknetConv2D_BN_Leaky(X, 512, (3, 3), strides=(1, 1), is_train=is_train, name='9')
        X = DarknetConv2D_BN_Leaky(X, 256, (1, 1), strides=(1, 1), is_train=is_train, name='10')
        X = DarknetConv2D_BN_Leaky(X, 512, (3, 3), strides=(1, 1), is_train=is_train, name='11')
        X = DarknetConv2D_BN_Leaky(X, 256, (1, 1), strides=(1, 1), is_train=is_train, name='12')
        X = DarknetConv2D_BN_Leaky(X, 512, (3, 3), strides=(1, 1), is_train=is_train, name='13')
        path_1 = LambdaLayer(X, fn=tf.space_to_depth,
                             fn_args={'block_size': 2,
                                        'name': 'path_1'},
                             name='path_1')
        X = MaxPool2d(X, (2, 2), padding='VALID', name='pool_5')

        X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1), is_train=is_train, name='14')
        X = DarknetConv2D_BN_Leaky(X, 512, (1, 1), strides=(1, 1), is_train=is_train, name='15')
        X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1), is_train=is_train, name='16')
        X = DarknetConv2D_BN_Leaky(X, 512, (1, 1), strides=(1, 1), is_train=is_train, name='17')
        X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1), is_train=is_train, name='18')

        darknet19 = DarknetConv2D_BN_Leaky(X, cfg.IMAGENET_CLASSCOUNT, (1, 1),
                                           strides=(1, 1), is_train=is_train, name='darknet_body')
        avg_kernel_size = cfg.IMAGENET_AVGPOOLSIZE
        darknet19 = MeanPool2d(darknet19, (avg_kernel_size, avg_kernel_size))
        darknet19 = ReshapeLayer(darknet19, shape=(-1, cfg.IMAGENET_CLASSCOUNT), name='reshape_darknet19')

        X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1), is_train=is_train, name='19')
        X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1), is_train=is_train, name='20')
        path_2 = X

        X = ConcatLayer([path_1, path_2], concat_dim=-1, name='concat2path')
        # print(X.outputs.shape)
        X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1), is_train=is_train, name='21')

        lab_filters = cfg.BOXES_PER_CELL*(5 + cfg.CLASSES_COUNT)
        X = DarknetConv2D_BN_Leaky(X, lab_filters, (1, 1), strides=(1, 1), is_train=is_train, name='22')
        X = ReshapeLayer(X, shape=(-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT), name='reshape')

        tl.ops.enable_print()

        # X.print_layers()
        # X.print_params()

    return X, darknet19


def predic(yolo_output):

    yolo_output = tf.constant(yolo_output, tf.float32)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(yolo_output)
    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.45)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    return scores.eval(), (boxes.eval()*cfg.IMAGE_SIZE).astype(np.int32), classes.eval()


def image_label_show(image, scores, boxes, classes, name='image', delay=500):

    image = image.copy()

    for i in range(len(boxes)):
        cv2.rectangle(image, (boxes[i][1], boxes[i][0]),
                      (boxes[i][3], boxes[i][2]),
                      (0, 0, 255), 2)

    cv2.imshow(name, image)
    cv2.waitKey(delay)


def darknent19_interface(imagenet_label, darknet19):
    L2_regular = 0
    for w in tl.layers.get_variables_with_name('W_conv2d', train_only=True, printable=True):  # [-3:]:
        print("yolo network L2:", w.name)
        L2_regular += tf.contrib.layers.l2_regularizer(5e-4)(w)
        if w.name.find("conv2d_darknet_body") != -1:
            break

    y = darknet19.outputs
    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=imagenet_label, logits=y))
    ce += L2_regular

    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), imagenet_label)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return ce, acc


def train_darknet19(images_ph, darknet19_train, darknet19_test):

    learning_rate = 0.001

    imagenet_label = tf.placeholder(tf.int32, shape=[None], name='imagenet_label')
    train_loss, train_acc = darknent19_interface(imagenet_label, darknet19_train)
    test_loss, test_acc = darknent19_interface(imagenet_label, darknet19_test)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(train_loss, var_list=yolo_network_train.all_params)

    tf.summary.scalar('darknet19_total_loss', test_loss)
    tf.summary.scalar('darknet19_acc', test_acc)



def main():
    sess = tf.InteractiveSession()

    epoch_start = 1
    learning_rate = 0.1

    images_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='image_data')
    matching_true_boxes = tf.placeholder(
        tf.float32, shape=[None, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT],
        name='matching_true_boxes')
    true_boxes = tf.placeholder(
        tf.float32, shape=[None, cfg.MAX_TRUEBOXS, 4], name='true_boxes')

    yolo_network_train, darknet19_train = yolo_build_model(images_ph, False, True)
    yolo_network_test, darknet19_test = yolo_build_model(images_ph, True, False)
    yolo_loss_train, _, _, _ = yolo_loss(yolo_network_train.outputs, matching_true_boxes, true_boxes)
    yolo_loss_test, confidence_loss_test, classification_loss_test, coordinates_loss_test= \
        yolo_loss(yolo_network_test.outputs, matching_true_boxes, true_boxes)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(yolo_loss_train, var_list=yolo_network_train.all_params)

    #darknet19
    imagenet_label = tf.placeholder(tf.int32, shape=[None], name='imagenet_label')
    train_darknet19_loss, train_darknet19_acc = darknent19_interface(imagenet_label, darknet19_train)
    test_darknet19_loss, test_darknet19_acc = darknent19_interface(imagenet_label, darknet19_test)
    train_darknet19_op = tf.train.AdamOptimizer(learning_rate).minimize(train_darknet19_loss, var_list=darknet19_train.all_params)

    tl.layers.initialize_global_variables(sess)

    yolo_network_train.print_params(False)
    yolo_network_train.print_layers()


    tf.summary.scalar('total_loss', yolo_loss_test)
    tf.summary.scalar('confidence_loss', confidence_loss_test)
    tf.summary.scalar('classification_loss', classification_loss_test)
    tf.summary.scalar('coordinates_loss', coordinates_loss_test)
    tf.summary.scalar('learning_rate', learning_rate)
    merged_tb = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(cfg.LOGS_TRAIN, sess.graph)
    test_writer = tf.summary.FileWriter(cfg.LOGS_TEST, sess.graph)

    if cfg.RESUME:
        print("Load existing model " + "!"*10)
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(cfg.MODEL_SAVE_DIR)
        if model_file is not None:
            epoch_start = epoch_start+int(model_file.split('-')[1])
            print(model_file)
            saver.restore(sess, model_file)

    if cfg.TRAIN_DARKNET19 == True:
        print('train start!!!!!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        for epoch in range(epoch_start, 2000):
            file_count = 0
            oldtime = datetime.datetime.now()
            while True:
                imgdata_path = os.path.join(cfg.IMAGENET_TRAINDATA,
                                 'data_' + str(cfg.IMAGENET_IMAGE_SIZE) + '_' + str(file_count) + '.npy')
                imglabel_path = os.path.join(cfg.IMAGENET_TRAINDATA,
                                  'label_' + str(cfg.IMAGENET_IMAGE_SIZE) + '_' + str(file_count) + '.npy')
                if os.path.isfile(imgdata_path) is True and os.path.isfile(imglabel_path) is True:
                    in_image_data = np.load(imgdata_path)
                    in_label_data = np.load(imglabel_path)

                    x_index = np.arange(in_image_data.shape[0])
                    y_index = np.arange(in_label_data.shape[0])

                    for x_index_a, y_index_a in tl.iterate.minibatches(
                            x_index, y_index, 12, shuffle=True):
                        sess.run([train_darknet19_op], feed_dict={images_ph: in_image_data[x_index_a],
                                                        imagenet_label: in_label_data[x_index_a]})
                    file_count += 1
                else:
                    break

            newtime = datetime.datetime.now()
            print("epoch:", epoch, "time_cost:", (newtime - oldtime).seconds,
                  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            if epoch % 1 == 0 and epoch != 0:
                saver = tf.train.Saver()
                saver.save(sess, os.path.join(cfg.MODEL_SAVE_DIR, cfg.MODEL_FILE_NAME), global_step=epoch)

    image_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.IMAGE_DATA))
    label_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.LABEL_DATA))
    trueboxs_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.TRUEBOX_DATA))

    split = int(image_data.shape[0] * 0.9)
    print(split, image_data.shape[0])

    train_data = image_data[:split]
    train_label = label_data[:split]
    train_trueboxs = trueboxs_data[:split]

    test_data = image_data[split:]
    test_label = label_data[split:]
    test_trueboxs = trueboxs_data[split:]

    data_size = 40
    start = 0
    end = start+data_size
    #
    # result_train = sess.run([yolo_network_test.outputs],
    #                                     feed_dict={images_ph: train_data[start:end],
    #                                                true_boxes: train_trueboxs[start:end],
    #                                                matching_true_boxes: train_label[start:end]})
    # result_train = result_train[0]
    #
    # result_test = sess.run([yolo_network_test.outputs],
    #                                   feed_dict={images_ph: test_data[start:end],
    #                                              true_boxes: test_trueboxs[start:end],
    #                                              matching_true_boxes: test_label[start:end]})
    # result_test = result_test[0]
    #
    # for i in range(start, end):
    #     scores, boxes, classes = predic(result_train[i-start])
    #     print(scores, boxes, classes)
    #     image_label_show(train_data[i], scores, boxes, classes, name='train'+str(i), delay=1)
    #
    #     scores, boxes, classes = predic(result_test[i-start])
    #     print(scores, boxes, classes)
    #     image_label_show(test_data[i], scores, boxes, classes, name='test'+str(i), delay=1)
    #
    # cv2.waitKey()

    print('train start!!!!!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(epoch_start, 2000):
        x_index = np.arange(train_data.shape[0])
        y_index = np.arange(train_label.shape[0])

        oldtime = datetime.datetime.now()
        for x_index_a, y_index_a in tl.iterate.minibatches(
                x_index, y_index, 11, shuffle=True):

            sess.run([train_op], feed_dict={images_ph: train_data[x_index_a],
                                          true_boxes: train_trueboxs[x_index_a],
                                          matching_true_boxes: train_label[x_index_a]})

        newtime = datetime.datetime.now()
        print("epoch:", epoch, "time_cost:", (newtime - oldtime).seconds, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

        if epoch % 1 == 0 and epoch != 0:
            saver = tf.train.Saver()
            save_path = saver.save(sess, os.path.join(cfg.MODEL_SAVE_DIR, cfg.MODEL_FILE_NAME), global_step=epoch)

        train_tb, total_loss_train, train_confidence_loss, train_classification_loss, train_coordinates_loss, result_train = \
            sess.run([merged_tb, yolo_loss_test, confidence_loss_test,
                      classification_loss_test, coordinates_loss_test,
                      yolo_network_test.outputs],
                     feed_dict={images_ph: train_data[start:end],
                                true_boxes: train_trueboxs[start:end],
                                matching_true_boxes: train_label[start:end]})
        print("epoch:", epoch, "train_total_loss:", total_loss_train,
              "train_conf_loss:", train_confidence_loss,
              "train_class_loss:", train_classification_loss,
              "train_coor_loss:", train_coordinates_loss)

        test_tb, test_total_loss, test_confidence_loss, test_classification_loss, test_coordinates_loss, result_test = \
            sess.run([merged_tb, yolo_loss_test, confidence_loss_test,
                      classification_loss_test, coordinates_loss_test,
                      yolo_network_test.outputs],
                     feed_dict={images_ph: test_data[start:end],
                                true_boxes: test_trueboxs[start:end],
                                matching_true_boxes: test_label[start:end]})
        print("epoch:", epoch, "test_total_loss :", test_total_loss,
              "test_conf_loss :", test_confidence_loss,
              "tset_class_loss :", test_classification_loss,
              "test_coor_loss :", test_coordinates_loss)

        train_writer.add_summary(train_tb, epoch)
        test_writer.add_summary(test_tb, epoch)

        scores, boxes, classes = predic(result_train[3])
        print(scores, boxes, classes)
        image_label_show(train_data[start+3], scores, boxes, classes, name='train')

        scores, boxes, classes = predic(result_test[3])
        print(scores, boxes, classes)
        image_label_show(test_data[start+3], scores, boxes, classes, name='test')



main()


















