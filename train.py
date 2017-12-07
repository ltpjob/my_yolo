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
    y_pred = K.reshape(y_pred, (-1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT))

    pred_box_confidence = y_pred[:, :, :, :, 0:1]
    pred_boxes = y_pred[:, :, :, :, 1:5]
    pred_box_class_probs = y_pred[:, :, :, :, 5:]

    true_box_confidence = y_true[:, :, :, :, 0:1]
    true_boxes = y_true[:, :, :, :, 1:5]
    true_box_class_probs = y_true[:, :, :, :, 5:]


    print(box_confidence.shape)
    print(boxes.shape)
    print(box_class_probs.shape)






def yolo_model_build(input_shape):
    X_input = Input(input_shape)
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




    model = Model(inputs=X_input, outputs=X, name='my_yolo')
    model.summary()




yolo_model_build((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3))

