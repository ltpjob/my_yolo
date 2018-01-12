import os
import tensorflow as tf
import trunk.config as cfg
from trunk.yolov2_loss import *


def DarknetConv2D_BN_Leaky(input, filters, kernel_size, strides=(1, 1), conv_kr=None, is_train=True, padding='same'):

    h_conv = tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                              kernel_regularizer=conv_kr)
    h_bn = tf.layers.batch_normalization(h_conv, training=is_train)
    h_act = tf.maximum(0.1 * h_bn, h_bn)

    return h_act


def darknet19_core(input, reuse=tf.AUTO_REUSE, is_train=True):
    with tf.variable_scope("darknet19_core", reuse=reuse):

        l2_regular = tf.contrib.layers.l2_regularizer(scale=5e-4)

        #416to208
        layer_0= DarknetConv2D_BN_Leaky(input, 32, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_1 = tf.layers.max_pooling2d(layer_0, pool_size=(2, 2), strides=(2, 2), padding='same')

        #208to104
        layer_2 = DarknetConv2D_BN_Leaky(layer_1, 64, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_3 = tf.layers.max_pooling2d(layer_2, pool_size=(2, 2), strides=(2, 2), padding='same')

        # 104to52
        layer_4 = DarknetConv2D_BN_Leaky(layer_3, 128, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_5 = DarknetConv2D_BN_Leaky(layer_4, 64, (1, 1), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_6 = DarknetConv2D_BN_Leaky(layer_5, 128, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_7 = tf.layers.max_pooling2d(layer_6, pool_size=(2, 2), strides=(2, 2), padding='same')

        #52to26
        layer_8 = DarknetConv2D_BN_Leaky(layer_7, 256, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_9 = DarknetConv2D_BN_Leaky(layer_8, 128, (1, 1), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_10 = DarknetConv2D_BN_Leaky(layer_9, 256, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_11 = tf.layers.max_pooling2d(layer_10, pool_size=(2, 2), strides=(2, 2), padding='same')

        #26to13
        layer_12 = DarknetConv2D_BN_Leaky(layer_11, 512, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_13 = DarknetConv2D_BN_Leaky(layer_12, 256, (1, 1), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_14 = DarknetConv2D_BN_Leaky(layer_13, 512, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_15 = DarknetConv2D_BN_Leaky(layer_14, 256, (1, 1), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_16 = DarknetConv2D_BN_Leaky(layer_15, 512, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)

        layer_17 = tf.layers.max_pooling2d(layer_16, pool_size=(2, 2), strides=(2, 2), padding='same')

        layer_18 = DarknetConv2D_BN_Leaky(layer_17, 1024, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_19 = DarknetConv2D_BN_Leaky(layer_18, 512, (1, 1), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_20 = DarknetConv2D_BN_Leaky(layer_19, 1024, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_21 = DarknetConv2D_BN_Leaky(layer_20, 512, (1, 1), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_22 = DarknetConv2D_BN_Leaky(layer_21, 1024, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)

        return layer_22, layer_16


def darnet19_body(layer_22, reuse=tf.AUTO_REUSE, is_train=True):
    with tf.variable_scope("darknet19_body", reuse=reuse):

        l2_regular = tf.contrib.layers.l2_regularizer(scale=5e-4)

        darknet19_cls = DarknetConv2D_BN_Leaky(layer_22, cfg.IMAGENET_CLASSCOUNT, (1, 1),
                                               strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        avg_k_size = cfg.IMAGENET_AVGPOOLSIZE
        avg_pool = tf.layers.average_pooling2d(darknet19_cls, pool_size=(avg_k_size, avg_k_size),
                                               strides=(avg_k_size, avg_k_size), padding='same')
        logits = tf.reshape(avg_pool, shape=(-1, cfg.IMAGENET_CLASSCOUNT))

        return logits


def yolov2_body(layer_22, layer_16, reuse=tf.AUTO_REUSE, is_train=True):
    with tf.variable_scope("yolov2_body", reuse=reuse):

        l2_regular = tf.contrib.layers.l2_regularizer(scale=5e-4)

        layer_23 = DarknetConv2D_BN_Leaky(layer_22, 1024, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_24 = DarknetConv2D_BN_Leaky(layer_23, 1024, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)

        layer_25 = layer_16
        layer_26 = tf.space_to_depth(layer_25, block_size=2)

        layer_27 = tf.concat([layer_26, layer_24], axis=-1)

        layer_28 = DarknetConv2D_BN_Leaky(layer_27, 1024, (3, 3), strides=(1, 1), conv_kr=l2_regular, is_train=is_train)

        lab_filters = cfg.BOXES_PER_CELL*(5 + cfg.CLASSES_COUNT)
        layer_29 = DarknetConv2D_BN_Leaky(layer_28, lab_filters, (1, 1),
                                          strides=(1, 1), conv_kr=l2_regular, is_train=is_train)
        layer_30 = tf.reshape(layer_29,
                              shape=(-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT))

        return layer_30


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


def predic(yolo_output):

    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(yolo_output)
    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.45)

    # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    boxes = tf.cast(boxes*cfg.IMAGE_SIZE, tf.int32)

    return scores, boxes, classes

