import numpy as np
import tensorflow as tf
import trunk.config as cfg

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

    return total_loss, confidence_loss_mean, classification_loss_mean, coordinates_loss_mean


