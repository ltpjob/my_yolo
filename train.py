from keras import backend as K
from keras.layers import Input, Add, Dense, Activation, LeakyReLU, BatchNormalization, Reshape, Conv2D, concatenate, MaxPooling2D, Flatten
from keras.models import Model, load_model
import trunk.config as cfg
import numpy as np
import os
import tensorflow as tf
# from matplotlib.pyplot import imshow
import time
import cv2


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
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
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


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.6):
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

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

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


def image_boxes_to_corners(label):

    label = label.copy()

    for i in range(cfg.CELL_SIZE):
        for j in range(cfg.CELL_SIZE):
            for k in range(cfg.BOXES_PER_CELL):
                x = (label[i, j, k, 1] + i)*(cfg.IMAGE_SIZE/cfg.CELL_SIZE)
                y = (label[i, j, k, 2] + j)*(cfg.IMAGE_SIZE/cfg.CELL_SIZE)
                w = label[i, j, k, 3]*(cfg.IMAGE_SIZE/cfg.CELL_SIZE)
                h = label[i, j, k, 4]*(cfg.IMAGE_SIZE/cfg.CELL_SIZE)
                label[i, j, k, 1] = int(y - (h / 2))    #ymin
                label[i, j, k, 2] = int(x - (w / 2))    #xmin
                label[i, j, k, 3] = int(y + (h / 2))    #ymax
                label[i, j, k, 4] = int(x + (w / 2))    #xmax

                # x = label[i, j, k, 1] * cfg.IMAGE_SIZE
                # y = label[i, j, k, 2] * cfg.IMAGE_SIZE
                # w = label[i, j, k, 3] * cfg.IMAGE_SIZE
                # h = label[i, j, k, 4] * cfg.IMAGE_SIZE
                #
                # label[i, j, k, 1] = int(y - (h / 2))    #ymin
                # label[i, j, k, 2] = int(x - (w / 2))    #xmin
                # label[i, j, k, 3] = int(y + (h / 2))    #ymax
                # label[i, j, k, 4] = int(x + (w / 2))    #xmax

    return label


def label_split(label):
    box_confidence = label[:, :, :, 0:1]
    boxes = label[:, :, :, 1:5]
    box_class_probs = label[:, :, :, 5:]

    return box_confidence, boxes, box_class_probs


def image_trueboxs_to_corners(trueboxs):

    boxes = np.zeros(trueboxs.shape)
    boxes[:, 0] = trueboxs[:, 0] - trueboxs[:, 2] / 2
    boxes[:, 1] = trueboxs[:, 1] + trueboxs[:, 3] / 2
    boxes[:, 2] = trueboxs[:, 0] + trueboxs[:, 2] / 2
    boxes[:, 3] = trueboxs[:, 1] - trueboxs[:, 3] / 2

    print(boxes)

    boxes = (boxes*cfg.IMAGE_SIZE).astype(int)

    return boxes

def image_truebox_show(image, trueboxs, name='image'):

    image = image.copy()

    boxes = image_trueboxs_to_corners(trueboxs)
    for box in boxes:
        cv2.rectangle(image, (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 0, 255), 2)

    cv2.imshow(name, image)
    cv2.waitKey(500)



def image_label_show(image, label, name='image'):

    image = image.copy()

    label = image_boxes_to_corners(label)
    box_confidence, box_boxes, box_class_probs = label_split(label)
    ph_bcon = tf.placeholder(tf.float32, shape=box_confidence.shape)
    ph_boxes = tf.placeholder(tf.float32, shape=box_boxes.shape)
    ph_bclass = tf.placeholder(tf.float32, shape=box_class_probs.shape)

    scores, boxes, classes = yolo_filter_boxes(ph_bcon, ph_boxes, ph_bclass, threshold=0.1)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)

    sess = K.get_session()
    out_scores, out_boxes, out_classes = sess.run(
                    [scores, boxes, classes],
                     feed_dict={
                         ph_bcon: box_confidence,
                         ph_boxes: box_boxes,
                         ph_bclass: box_class_probs,
                         K.learning_phase(): 0
                     })

    if len(out_boxes) > 0:
        print("max!!!", max(out_scores))
    for i in range(len(out_boxes)):
        cv2.rectangle(image, (out_boxes[i][1], out_boxes[i][0]),
                      (out_boxes[i][3], out_boxes[i][2]),
                      (0, 0, 255), 2)

    cv2.imshow(name, image)
    cv2.waitKey(500)


def DarknetConv2D_BN_Leaky(X, filters, kernel_size, strides=(1, 1), padding='same'):

    X = Conv2D(filters, kernel_size, strides=strides, padding=padding)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)

    return X





def yolo_loss(y_true, y_pred):

    object_scale = cfg.OBJECT_SCALE
    noobject_scale = cfg.NOOBJECT_SCALE
    class_scale = cfg.CLASS_SCALE
    coord_scale = cfg.COORD_SCALE

    # y_pred = K.reshape(y_pred, (-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT))
    # y_true = K.reshape(y_true, (-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT))

    true_confidence = y_true[:, :, :, :, 0:1]
    true_boxes = y_true[:, :, :, :, 1:5]
    true_class_probs = y_true[:, :, :, :, 5:]

    pred_confidence = y_pred[:, :, :, :, 0:1]
    pred_boxes = y_pred[:, :, :, :, 1:5]
    pred_class_probs = y_pred[:, :, :, :, 5:]
    # print(true_class_probs.shape, pred_class_probs.shape)

    true_xy = true_boxes[:, :, :, :, 0:2]
    true_wh = true_boxes[:, :, :, :, 2:]
    pred_xy = pred_boxes[:, :, :, :, 0:2]
    pred_wh = pred_boxes[:, :, :, :, 2:]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas
    iou_scores = tf.expand_dims(iou_scores, 4)

    object_mask = K.cast(K.equal(true_confidence, 1), 'float32')
    noobject_mask = K.cast(K.equal(true_confidence, 0), 'float32')

    boxs_coord_pos = K.square(true_boxes[:, :, :, :, 0:1] - pred_boxes[:, :, :, :, 0:1]) \
                     + K.square(true_boxes[:, :, :, :, 1:2] - pred_boxes[:, :, :, :, 1:2])
    coord_pos_loss = coord_scale * K.sum(boxs_coord_pos * object_mask, axis=(1, 2, 3, 4))
    # print(boxs_coord_pos.shape, (boxs_coord_pos * object_mask).shape, coord_pos_loss.shape)

    boxs_coord_size = K.square(K.sqrt(true_boxes[:, :, :, :, 2:3]) - K.sqrt(pred_boxes[:, :, :, :, 2:3])) \
                     + K.square(K.sqrt(true_boxes[:, :, :, :, 3:4]) - K.sqrt(pred_boxes[:, :, :, :, 3:4]))
    coord_size_loss = coord_scale * K.sum(boxs_coord_size * object_mask, axis=(1, 2, 3, 4))
    # print(boxs_coord_size.shape, (boxs_coord_size * object_mask).shape, coord_size_loss.shape)

    obj_confidence_loss = object_scale * K.sum(object_mask * K.square(true_confidence - pred_confidence*iou_scores), axis=(1, 2, 3, 4))
    # print((object_mask * K.square(true_confidence - pred_confidence)).shape, obj_confidence_loss.shape)

    noobj_confidence_loss = noobject_scale * K.sum(noobject_mask * K.square(true_confidence - pred_confidence*iou_scores), axis=(1, 2, 3, 4))
    # print((noobject_mask * K.square(true_confidence - pred_confidence)).shape, noobj_confidence_loss.shape)

    boxes_class = K.sum(K.square(true_class_probs - pred_class_probs), axis=-1, keepdims=True)
    class_loss = class_scale * K.sum(object_mask * boxes_class, axis=(1, 2, 3, 4))
    # print(boxes_class.shape, class_loss.shape)

    loss = K.mean(coord_pos_loss + coord_size_loss + obj_confidence_loss + noobj_confidence_loss + class_loss)
    # print(loss.shape)

    return loss

def yolo_build_model(data_input_shape):
    X_input = Input(data_input_shape)
    #416to208
    X = DarknetConv2D_BN_Leaky(X_input, 32, (3, 3), strides=(1, 1))
    X = MaxPooling2D((2, 2))(X)

    #208to104
    X = DarknetConv2D_BN_Leaky(X, 64, (3, 3), strides=(1, 1))
    X = MaxPooling2D((2, 2))(X)

    X = DarknetConv2D_BN_Leaky(X, 128, (3, 3), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 64, (1, 1), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 128, (3, 3), strides=(1, 1))
    X = MaxPooling2D((2, 2))(X)

    X = DarknetConv2D_BN_Leaky(X, 256, (3, 3), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 128, (1, 1), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 256, (3, 3), strides=(1, 1))
    X = MaxPooling2D((2, 2))(X)

    X = DarknetConv2D_BN_Leaky(X, 512, (3, 3), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 256, (1, 1), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 512, (3, 3), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 256, (1, 1), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 512, (3, 3), strides=(1, 1))
    X = MaxPooling2D((2, 2))(X)

    X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 512, (1, 1), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 512, (1, 1), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1))

    darknet = Conv2D(1000, (1, 1), activation="softmax")(X)
    print(darknet.shape)

    X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1))
    lab_filters = cfg.BOXES_PER_CELL*(5 + cfg.CLASSES_COUNT)

    X = Conv2D(lab_filters, (1, 1), strides=(1, 1), padding='same')(X)
    X = Reshape((cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT))(X)
    # print(X.shape)

    model = Model(inputs=X_input, outputs=X, name='my_yolo')
    # model.summary()

    return model


def main():
    image_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.IMAGE_DATA))
    label_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.LABEL_DATA))
    trueboxs_data = np.load(os.path.join(cfg.TRAIN_DATA_DIR, cfg.TRUEBOX_DATA))


    split = int(image_data.shape[0] * 0.7)
    print(split, image_data.shape[0])

    train_data = image_data[:split]
    train_label = label_data[:split]

    test_data = image_data[split:]
    test_label = label_data[split:]
    K.dtype()

    # imshow(test_data[0])
    # mat_array = cv2.fromarray(test_data[0])

    start = 0
    end = 3

    for i in range(start, end):
        image_truebox_show(train_data[i], trueboxs_data[i], "image"+str(i))
        print(train_label[i][7][7])

    # image_label_show(train_data[13], train_label[13])
    # image_label_show(train_data[10], train_label[10])
    # # print(np.where(train_label[13]==1))
    # print(train_label[10][7][7])

    # cv2.imshow("test", test_data[0])

    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)

    model = yolo_build_model((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3))
    model.compile(optimizer='adam', loss=yolo_loss)

    for i in range(100):
        model.fit(x=train_data/255, y=train_label, batch_size=8)
        # model.train_on_batch(x=train_data[0:20], y=train_label[0:20])
        preds = model.evaluate(test_data/255, test_label)
        print(i, "Loss = " + str(preds))


        # print(pred.shape, pred[0].shape)
        # print(pred)

        pred = model.predict(train_data[start:end]/255)
        for j in range(start, end):
            print(pred[j-start][7][7])
            print(pred[j-start][0][0])
            cv2.destroyWindow('train_' + str(i-1)+str(j))
            image_label_show(train_data[j], pred[j-start], 'train_' + str(i)+str(j))

    cv2.waitKey(0)




main()



