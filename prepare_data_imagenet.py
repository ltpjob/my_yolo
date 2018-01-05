import os
import trunk.config as cfg
import shutil
import tarfile
import cv2
import numpy as np


def un_tar(file_name, extdir):
    tar = tarfile.open(file_name)  
    names = tar.getnames()  
    if os.path.isdir(extdir):
        pass  
    else:  
        os.mkdir(extdir)

    for name in names:  
        tar.extract(name, extdir)
    tar.close()


def unpack_imagenet_tar(class_name):
    for key, value in class_name.items():
        tar_src = os.path.join(cfg.IMAGENET_TRAINIMAGE_TAR, key) + '.tar'
        unpack_dst = os.path.join(cfg.IMAGENET_TRAINIMAGE, key)
        if os.path.isfile(tar_src) is True:
            un_tar(tar_src, unpack_dst)
        else:
            print(tar_src, 'is not found!!!')


def train_data_make(class_name, class_index):
    count = 0
    save_count = 0
    image_data = []
    label_data = []
    for key, value in class_name.items():
        imagedir = os.path.join(cfg.IMAGENET_TRAINIMAGE, key)
        for parent, dirnames, filenames in os.walk(imagedir):
            for filename in filenames:
                headname, expname = os.path.splitext(filename)
                expname = expname.lower()
                image_id = headname.split('_')[0]
                full_path = os.path.join(imagedir, filename)
                # str11 = value#str(value).encode('utf-8')
                # print(str11)
                # return

                if expname == '.jpeg' and image_id == key:
                    count += 1
                    print(full_path, class_index[key], count)
                    img = cv2.imread(full_path)
                    img = cv2.resize(img, (cfg.IMAGENET_IMAGE_SIZE, cfg.IMAGENET_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                    image_data.append(img)
                    label_data.append(class_index[key])
                    print(len(image_data))
                    if len(image_data) == 20000:
                        data_path =os.path.join(cfg.IMAGENET_TRAINDATA, 'data_'+str(cfg.IMAGENET_IMAGE_SIZE)+'_'+str(save_count)+'.npy')
                        label_path = os.path.join(cfg.IMAGENET_TRAINDATA, 'label_'+str(cfg.IMAGENET_IMAGE_SIZE)+'_'+str(save_count)+'.npy')
                        data_np = np.array(image_data)
                        label_np = np.array(label_data)
                        np.save(data_path, data_np)
                        np.save(label_path, label_np)
                        image_data = []
                        label_data = []
                        save_count += 1

    if len(image_data) > 0:
        data_path = os.path.join(cfg.IMAGENET_TRAINDATA,
                                 'data_' + str(cfg.IMAGENET_IMAGE_SIZE) + '_' + str(save_count) + '.npy')
        label_path = os.path.join(cfg.IMAGENET_TRAINDATA,
                                  'label_' + str(cfg.IMAGENET_IMAGE_SIZE) + '_' + str(save_count) + '.npy')
        data_np = np.array(image_data)
        label_np = np.array(label_data)
        np.save(data_path, data_np)
        np.save(label_path, label_np)





def main():
    class_name, class_index = cfg.pickup_dic()

    # unpack_imagenet_tar(class_name)

    # train_data_make(class_name, class_index)






main()