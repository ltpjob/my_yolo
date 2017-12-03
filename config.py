VOCdevkit_dir = 'D:/project/VOCdevkit/'
VOC2012_dir = VOCdevkit_dir + "VOC2012/"
VOC2007_dir = VOCdevkit_dir + "VOC2007/"
root_dir = VOC2007_dir
pic_dir = root_dir + "JPEGImages/"
Annotations_dir = root_dir + "Annotations/"

CONVERT_DATA_dir = './data'
TRAIN_DATA_DIR = "./train_data"

CLASSES = ['car']

CLASSES_COUNT = len(CLASSES)

IMAGE_SIZE = 416

CELL_SIZE = 13

ANCHOR_BOXES = [(200, 140), (66, 266)]

BOXES_PER_CELL = len(ANCHOR_BOXES)

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0