VOCdevkit_dir = 'D:/project/VOCdevkit/'
VOC2012_dir = VOCdevkit_dir + "VOC2012/"
VOC2007_dir = VOCdevkit_dir + "VOC2007/"
root_dir = VOC2012_dir
pic_dir = root_dir + "JPEGImages/"
Annotations_dir = root_dir + "Annotations/"

CONVERT_DATA_dir = './data'
TRAIN_DATA_DIR = "./train_data"
MODEL_SAVE_DIR = "./model_save"

CLASSES = ['person', 'aeroplane', 'tvmonitor', 'train', 'boat', 'dog', 'chair', 'bird', 'bicycle', 'bottle', 'sheep', 'diningtable', 'horse', 'motorbike', 'sofa', 'cow', 'car', 'cat', 'bus', 'pottedplant']

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
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1
COORD_SCALE = 5

IMAGE_DATA = "image_data.npy"
LABEL_DATA = "label_data.npy"
TRUEBOX_DATA = "truebox_data.npy"


MODEL_FILE_NAME = "model_my_yolo.ckpt"
RESUME = True # load model, resume from previous checkpoint?

LOGS_TRAIN = "logs/train_my_yolo_1"
LOGS_TEST = "logs/test_my_yolo_1"