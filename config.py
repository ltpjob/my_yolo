import os
VOCdevkit_dir = 'D:/project/VOCdevkit/'
VOC2012_dir = VOCdevkit_dir + "VOC2012/"
VOC2007_dir = VOCdevkit_dir + "VOC2007/"
root_dir = VOC2007_dir
pic_dir = root_dir + "JPEGImages/"
Annotations_dir = root_dir + "Annotations/"

CONVERT_DATA_dir = './data'
TRAIN_DATA_DIR = "./train_data"
MODEL_SAVE_DIR = "./model_save"
TEST_MODEL_SAVE_DIR = "./test_model_save"
MODEL_DIR_ROOT = 'F:\\'
DARKNET19_MODEL_SAVE_DIR = os.path.join(MODEL_DIR_ROOT, "darnet19_model_save")
YOLOV2_MODEL_SAVE_DIR = os.path.join(MODEL_DIR_ROOT, "yolov2_model_save")


CLASSES = ['dog', 'person', 'train', 'sofa', 'chair', 'car', 'pottedplant', 'diningtable', 'horse', 'cat', 'cow', 'bus', 'bicycle', 'aeroplane', 'motorbike', 'tvmonitor', 'bird', 'bottle', 'boat', 'sheep']

CLASSES_COUNT = len(CLASSES)

MAX_TRUEBOXS = 40

IMAGE_SIZE = 416

CELL_SIZE = 13

# ANCHOR_BOXES = [(200, 140), (66, 266)]

# ANCHOR_BOXES = [(6.25, 4.375), (2.0625, 8.3125)]

ANCHOR_BOXES = [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]

BOXES_PER_CELL = len(ANCHOR_BOXES)

ALPHA = 0.1

DISP_CONSOLE = False

# OBJECT_SCALE = 0
# NOOBJECT_SCALE = 0
# CLASS_SCALE = 0
# COORD_SCALE = 1

OBJECT_SCALE = 1
NOOBJECT_SCALE = 1
CLASS_SCALE = 1
COORD_SCALE = 5

IMAGE_DATA = "image_data.npy"
LABEL_DATA = "label_data.npy"
TRUEBOX_DATA = "truebox_data.npy"


YOLOV2_MODEL_FILE_NAME = "model_my_yolo.ckpt"
DARKNET19_MODEL_FILE_NAME = "model_my_darknet19.ckpt"
RESUME = True # load model, resume from previous checkpoint?
TRAIN_DARKNET19 = True
RESUME_DARKNET19_CORE = True
RESUME_DARKNET19_TRAIN = True
RESUME_YOLOV2_TRAIN = True

LOGS_TRAIN = "logs/train_my_yolo_8"
LOGS_TEST = "logs/test_my_yolo_8"


IMAGENET_DATA_DIC = os.path.join(TRAIN_DATA_DIR, "imagenet_data_dic.pkl")
IMAGENET_PATH = "E:/image_net"
IMAGENET_TRAINIMAGE_TAR = os.path.join(IMAGENET_PATH, "train_image_tar")
IMAGENET_TRAINIMAGE = os.path.join(IMAGENET_PATH, "train_image")
IMAGENET_LABEL = os.path.join(IMAGENET_PATH, "ILSVRC2012_bbox_train_v2")
IMAGENET_CLASS_PICKUP = os.path.join(IMAGENET_PATH, "class_pickup.txt")
IMAGENET_TRAINDATA= os.path.join(IMAGENET_PATH, "train_data")
IMAGENET_IMAGE_SIZE = 224
IMAGENET_AVGPOOLSIZE = IMAGENET_IMAGE_SIZE/32


def pickup_dic():
    class_name = {}
    class_index = {}
    if os.path.isfile(IMAGENET_CLASS_PICKUP):
        pickup = open(IMAGENET_CLASS_PICKUP, mode='r', encoding='utf-8')
        index = 0
        lines = pickup.readlines()
        for line in lines:
            if len(line) > 9:
                strings = line.split()
                class_name[strings[0]] = strings[1]
                class_index[strings[0]] = index
                index += 1
        # print(class_name)
        # print(class_index)

    return class_name, class_index


IMAGENET_CLASS_NAME, _ = pickup_dic()
IMAGENET_CLASSCOUNT = len(IMAGENET_CLASS_NAME)


