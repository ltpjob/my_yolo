from keras import backend as K
from keras.layers import Input, Add, Dense, Activation, LeakyReLU, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
import trunk.config as cfg


def DarknetConv2D_BN_Leaky(X,   filters, kernel_size, strides=(1, 1), padding='same'):

    X = Conv2D(filters, kernel_size, strides=strides, padding=padding)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)

    return X


def yolo_loss(y_true, y_pred):

    object_scale = cfg.OBJECT_SCALE
    noobject_scale = cfg.NOOBJECT_SCALE
    class_scale = cfg.CLASS_SCALE
    coord_scale = cfg.COORD_SCALE

    y_pred = K.reshape(y_pred, (-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT))
    y_true = K.reshape(y_true, (-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT))

    true_confidence = y_true[:, :, :, :, 0:1]
    true_boxes = y_true[:, :, :, :, 1:5]
    true_class_probs = y_true[:, :, :, :, 5:]

    pred_confidence = y_pred[:, :, :, :, 0:1]
    pred_boxes = y_pred[:, :, :, :, 1:5]
    pred_class_probs = y_pred[:, :, :, :, 5:]
    # print(true_class_probs.shape, pred_class_probs.shape)

    object_mask = K.cast(K.equal(true_confidence, 1), 'float32')
    noobject_mask = K.cast(K.equal(true_confidence, 0), 'float32')

    boxs_coord_pos = K.square(true_boxes[:, :, :, :, 0:1] - pred_boxes[:, :, :, :, 0:1]) \
                     + K.square(true_boxes[:, :, :, :, 1:2] - pred_boxes[:, :, :, :, 1:2])
    coord_pos_loss = coord_scale * K.mean(boxs_coord_pos * object_mask, axis=(1, 2, 3, 4))
    # print(boxs_coord_pos.shape, (boxs_coord_pos * object_mask).shape, coord_pos_loss.shape)

    boxs_coord_size = K.square(K.sqrt(true_boxes[:, :, :, :, 2:3]) - K.sqrt(pred_boxes[:, :, :, :, 2:3])) \
                     + K.square(K.sqrt(true_boxes[:, :, :, :, 3:4]) - K.sqrt(pred_boxes[:, :, :, :, 3:4]))
    coord_size_loss = coord_scale * K.mean(boxs_coord_size * object_mask, axis=(1, 2, 3, 4))
    # print(boxs_coord_size.shape, (boxs_coord_size * object_mask).shape, coord_size_loss.shape)

    obj_confidence_loss = object_scale * K.mean(object_mask * K.square(true_confidence - pred_confidence), axis=(1, 2, 3, 4))
    # print((object_mask * K.square(true_confidence - pred_confidence)).shape, obj_confidence_loss.shape)

    noobj_confidence_loss = noobject_scale * K.mean(noobject_mask * K.square(true_confidence - pred_confidence), axis=(1, 2, 3, 4))
    # print((noobject_mask * K.square(true_confidence - pred_confidence)).shape, noobj_confidence_loss.shape)

    boxes_class = K.mean(K.square(true_class_probs - pred_class_probs), axis=-1, keepdims=True)
    class_loss = class_scale * K.mean(object_mask * boxes_class, axis=(1, 2, 3, 4))
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
    X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1))
    X = DarknetConv2D_BN_Leaky(X, 1024, (3, 3), strides=(1, 1))
    lab_filters = cfg.BOXES_PER_CELL*(5 + cfg.CLASSES_COUNT)

    X = Conv2D(lab_filters, (1, 1), strides=(1, 1), padding='same')(X)

    print(X.shape)

    # print(pred_confidence.shape, true_confidence.shape)
    # print(pred_boxes.shape, true_boxes.shape)
    # print(pred_class_probs.shape, true_class_probs.shape)

    model = Model(inputs=X_input, outputs=X, name='my_yolo')
    model.summary()

    return model


def main():

    model = yolo_build_model((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3))
    model.compile(optimizer='adam', loss=yolo_loss)


main()



