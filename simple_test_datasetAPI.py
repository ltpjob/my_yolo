import tensorflow as tf
import trunk.config as cfg
import os



# def _parse_function(filename):
#     image_string = tf.read_file(filename)
#     image_decoded = tf.image.decode_jpeg(image_string)
#     image_resized = tf.image.resize_images(image_decoded, [224, 224])
#     headname = os.path.splitext(os.path.split(filename)[1])[0]
#     id = headname.split("_")[0]
#     label = class_index[id]
#     return image_resized

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    return image_resized, label


def main():
    class_name, class_index = cfg.pickup_dic()

    image_filenames = []
    image_label = []
    for key, value in class_name.items():
        imagedir = os.path.join(cfg.IMAGENET_TRAINIMAGE, key)
        for parent, dirnames, filenames in os.walk(imagedir):
            for filename in filenames:
                headname, expname = os.path.splitext(filename)
                expname = expname.lower()
                image_id = headname.split('_')[0]
                full_path = os.path.join(imagedir, filename)

                if expname == '.jpeg' and image_id == key:
                    image_filenames.append(full_path)
                    image_label.append(class_index[image_id])
                    print(full_path, len(image_filenames), len(image_label))

    cons_filenames = tf.constant(image_filenames)
    cons_labels = tf.constant(image_label)
    dataset = tf.data.Dataset.from_tensor_slices((cons_filenames, cons_labels))
    dataset = dataset.map(_parse_function)

    dataset = dataset.shuffle(buffer_size=10000).batch(32).repeat(1)

    iterator = dataset.make_one_shot_iterator()
    next_example, next_label = iterator.get_next()


    with tf.Session() as sess:
        for i in range(10000):
            value = sess.run(dataset)
            print(type(value))
            # print(value[0].shape, value[1].shape)
    #
    # filenames = tf.constant(image_filenames)
    # # label[i]就是图片filenames[i]的label
    #
    # # 此时dataset中的一个元素是(filename, label)
    # dataset = tf.data.Dataset.from_tensor_slices(filenames)
    #
    # dataset = dataset.map(_parse_function)


main()