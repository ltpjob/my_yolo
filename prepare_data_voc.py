import os
import trunk.config as cfg
from xml.dom.minidom import parse
import xml.dom.minidom
from PIL import Image
import cv2
import numpy as np
from scipy import ndimage



def data_convert(class_filter):
    for parent, dirnames, filenames in os.walk(cfg.Annotations_dir):
        for filename in filenames:
            print("parent folder is:" + parent)
            print("filename with full path:" + os.path.join(parent, filename))
            DOMTree = xml.dom.minidom.parse(os.path.join(parent, filename))
            Data = DOMTree.documentElement
            pic_size = Data.getElementsByTagName("size")
            pic_wscale = float(pic_size[0].getElementsByTagName('width')[0].childNodes[0].nodeValue)/cfg.IMAGE_SIZE
            pic_hscale = float(pic_size[0].getElementsByTagName('height')[0].childNodes[0].nodeValue)/cfg.IMAGE_SIZE
            print(pic_wscale, pic_hscale)
            pic_size[0].getElementsByTagName('width')[0].childNodes[0].nodeValue = cfg.IMAGE_SIZE
            pic_size[0].getElementsByTagName('height')[0].childNodes[0].nodeValue = cfg.IMAGE_SIZE

            objects = Data.getElementsByTagName("object")
            flag_save = 0
            for obj in objects:
                obj_name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
                if obj_name not in class_filter:
                    Data.removeChild(obj)
                    continue
                else:
                    flag_save += 1

                obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue = \
                    int(round(int(obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)/pic_wscale))
                obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue = \
                    int(round(int(obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)/pic_wscale))
                obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue = \
                    int(round(int(obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)/pic_hscale))
                obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue = \
                    int(round(int(obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)/pic_hscale))

            if flag_save == 0:
                continue

            jpeg_filename = Data.getElementsByTagName("filename")[0].childNodes[0].nodeValue
            im = Image.open(os.path.join(cfg.pic_dir, jpeg_filename))
            out = im.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), Image.ANTIALIAS)
            out.save(os.path.join(cfg.CONVERT_DATA_dir, jpeg_filename))

            f = open(os.path.join(cfg.CONVERT_DATA_dir, filename), 'w', encoding='utf-8')
            DOMTree.writexml(f, encoding='utf-8')
            f.close()


def image_show():
    for parent, dirnames, filenames in os.walk(cfg.CONVERT_DATA_dir):
        for filename in filenames:
            # print("parent folder is:" + parent)
            # print("filename with full path:" + os.path.join(parent, filename))
            if os.path.splitext(filename)[1] == '.xml':
                DOMTree = xml.dom.minidom.parse(os.path.join(parent, filename))
                Data = DOMTree.documentElement
                pic_size = Data.getElementsByTagName("size")
                pic_width = int(pic_size[0].getElementsByTagName('width')[0].childNodes[0].nodeValue)
                pic_height = int(pic_size[0].getElementsByTagName('height')[0].childNodes[0].nodeValue)
                # print(pic_width, pic_height)
                jpeg_filename = Data.getElementsByTagName("filename")[0].childNodes[0].nodeValue
                cell_width = int(pic_width/cfg.CELL_SIZE)
                cell_height = int(pic_height/cfg.CELL_SIZE)

                img = cv2.imread(os.path.join(parent, jpeg_filename))

                for i in range(1, cfg.CELL_SIZE):
                    cell_color = (129, 129, 129)
                    cv2.line(img, (i * cell_width, 0), (i * cell_width, pic_height), cell_color, 2)
                    cv2.line(img, (0, i * cell_height), (pic_width, i * cell_height), cell_color, 2)

                objects = Data.getElementsByTagName("object")
                for obj in objects:
                    obj_xmin = int(obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
                    obj_xmax = int(obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
                    obj_ymin = int(obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
                    obj_ymax = int(obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
                    obj_name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
                    obj_center = (round(obj_xmin+(obj_xmax-obj_xmin)/2), round(obj_ymin+(obj_ymax-obj_ymin)/2))
                    box_pos = (int(obj_center[0]*cfg.CELL_SIZE/pic_width),
                               int(obj_center[1]*cfg.CELL_SIZE/pic_height))
                    iou_list = []
                    for i in range(cfg.BOXES_PER_CELL):
                        anchor_box = (obj_center[0] - cfg.ANCHOR_BOXES[i][0]/2,
                                      obj_center[1] - cfg.ANCHOR_BOXES[i][1]/2,
                                      obj_center[0] + cfg.ANCHOR_BOXES[i][0]/2,
                                      obj_center[1] + cfg.ANCHOR_BOXES[i][1]/2,
                                      )
                        cv2.rectangle(img, (int(anchor_box[0]), int(anchor_box[1])),
                                      (int(anchor_box[2]), int(anchor_box[3])),
                                      (128, 0, 128), 2)
                        obj_box = (obj_xmin, obj_ymin, obj_xmax, obj_ymax)
                        iou = calc_iou(anchor_box, obj_box)
                        iou_list.append(iou)

                    #确定位置
                    cellpos_bx = (obj_center[0] % cell_width)/cell_width
                    cellpos_by = (obj_center[1] % cell_height)/cell_height
                    cellpos_bw = ((obj_xmax-obj_xmin)/cell_width)
                    cellpos_bh = ((obj_ymax-obj_ymin)/cell_height)
                    obj_pos = [1, cellpos_bx, cellpos_by, cellpos_bw, cellpos_bh]

                    #确定类别
                    obj_class = np.zeros(cfg.CLASSES_COUNT)
                    obj_class[cfg.CLASSES.index(obj_name)] = 1

                    obj_lab = np.append(obj_pos, obj_class)

                    best_box = np.array(iou_list).argmax()
                    print(box_pos, best_box, obj_lab)

                    cv2.circle(img, obj_center, 2, (255, 0, 0), 2)
                    cv2.rectangle(img, (obj_xmin, obj_ymin), (obj_xmax, obj_ymax), (0, 0, 255), 2)

                cv2.imshow(jpeg_filename, img)
                cv2.waitKey(0)


def calc_iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (yi2 - yi1) * (xi2 - xi1)
    ### END CODE HERE ###

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
    ### END CODE HERE ###

    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = inter_area / union_area
    ### END CODE HERE ###

    return iou

def data_make():

    image_data = []
    label_data = []
    trueboxs_data = []
    anchors = np.array(cfg.ANCHOR_BOXES)
    for parent, dirnames, filenames in os.walk(cfg.CONVERT_DATA_dir):
        for filename in filenames:
            # print("parent folder is:" + parent)
            # print("filename with full path:" + os.path.join(parent, filename))
            if os.path.splitext(filename)[1] == '.xml':
                DOMTree = xml.dom.minidom.parse(os.path.join(parent, filename))
                Data = DOMTree.documentElement
                pic_size = Data.getElementsByTagName("size")
                pic_width = int(pic_size[0].getElementsByTagName('width')[0].childNodes[0].nodeValue)
                pic_height = int(pic_size[0].getElementsByTagName('height')[0].childNodes[0].nodeValue)
                jpeg_filename = Data.getElementsByTagName("filename")[0].childNodes[0].nodeValue
                cell_width = int(pic_width/cfg.CELL_SIZE)
                cell_height = int(pic_height/cfg.CELL_SIZE)

                image_label = np.zeros((cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 5 + cfg.CLASSES_COUNT),
                                       dtype=np.float32)
                trueboxs = np.zeros((cfg.MAX_TRUEBOXS, 4))

                objects = Data.getElementsByTagName("object")

                for k, obj in enumerate(objects):
                    if k >= cfg.MAX_TRUEBOXS:
                        break

                    obj_xmin = int(obj.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
                    obj_xmax = int(obj.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
                    obj_ymin = int(obj.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
                    obj_ymax = int(obj.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
                    obj_name = obj.getElementsByTagName('name')[0].childNodes[0].nodeValue
                    obj_center = (obj_xmin+(obj_xmax-obj_xmin)/2, obj_ymin+(obj_ymax-obj_ymin)/2)
                    # 计算obj中点在哪个cell里
                    box_cellpos = (int(obj_center[0]*cfg.CELL_SIZE/pic_width),
                               int(obj_center[1]*cfg.CELL_SIZE/pic_height))

                    box_bx = (obj_xmin+(obj_xmax-obj_xmin)/2)/pic_width
                    box_by = (obj_ymin+(obj_ymax-obj_ymin)/2)/pic_height
                    box_bw = ((obj_xmax-obj_xmin)/pic_width)
                    box_bh = ((obj_ymax-obj_ymin)/pic_height)
                    truebox = [box_bx, box_by, box_bw, box_bh]
                    # print(truebox)

                    #确定位置
                    cellpos_bx = (obj_center[0] % cell_width)/cell_width
                    cellpos_by = (obj_center[1] % cell_height)/cell_height
                    cellpos_bw = ((obj_xmax-obj_xmin)/cell_width)
                    cellpos_bh = ((obj_ymax-obj_ymin)/cell_height)

                    obj_pos = [1, cellpos_bx, cellpos_by, cellpos_bw, cellpos_bh]

                    #确定类别
                    obj_class = np.zeros(cfg.CLASSES_COUNT)
                    obj_class[cfg.CLASSES.index(obj_name)] = 1

                    obj_lab = np.append(obj_pos, obj_class)

                    #找出iou最大的anchor box
                    iou_list = []
                    for anchor in anchors:
                        box_maxes = obj_lab[3:5] / 2.
                        box_mins = -box_maxes
                        anchor_maxes = (anchor / 2.)
                        anchor_mins = -anchor_maxes

                        intersect_mins = np.maximum(box_mins, anchor_mins)
                        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                        intersect_area = intersect_wh[0] * intersect_wh[1]

                        box_area = obj_lab[3] * obj_lab[4]
                        anchor_area = anchor[0] * anchor[1]
                        iou = intersect_area / (box_area + anchor_area - intersect_area)
                        iou_list.append(iou)

                    best_box = np.array(iou_list).argmax()
                    # print(best_box)

                    if image_label[box_cellpos[0], box_cellpos[1], best_box, 0] == 1:
                        print(jpeg_filename, image_label[box_cellpos[0], box_cellpos[1], best_box])
                        continue

                    obj_lab[3] = np.log(obj_lab[3] / anchors[best_box][0])
                    obj_lab[4] = np.log(obj_lab[4] / anchors[best_box][1])
                    image_label[box_cellpos[0], box_cellpos[1], best_box] = obj_lab
                    trueboxs[k] = truebox
                    # print(best_box, obj_lab)
                    # print(image_label[box_cellpos[0], box_cellpos[1], best_box, 0])

                image = np.array(cv2.imread(os.path.join(parent, jpeg_filename), cv2.IMREAD_COLOR))
                # print(image.shape)
                image_data.append(image)
                label_data.append(image_label)
                trueboxs_data.append(trueboxs)

    np.save(os.path.join(cfg.TRAIN_DATA_DIR, cfg.IMAGE_DATA), np.array(image_data))
    np.save(os.path.join(cfg.TRAIN_DATA_DIR, cfg.LABEL_DATA), np.array(label_data))
    np.save(os.path.join(cfg.TRAIN_DATA_DIR, cfg.TRUEBOX_DATA), np.array(trueboxs_data))




# data_convert(cfg.CLASSES)
# image_show()
data_make()


