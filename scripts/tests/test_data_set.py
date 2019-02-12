import os
import sys
import cv2
import shutil
from tensorflow.examples.tutorials.mnist import input_data

sys.path.append('../')
from TFUtils.DataSets import ClassificationDataSet

def test_data_set(data_dir):

    print('-------------------------------------------------------------------')
    print('Extracting Images into directory structure')
    cls_dataset = ClassificationDataSet(
                                data_set_name='Fashion MNIST',
                                data_set_description='Dataset created during testing of data set')
    cls_dataset.create_dir_structure(os.path.join(data_dir, cls_dataset.data_set_name))
    cls_data_dir = extract_classification_images(cls_dataset.data_set_dir+'/images/')
    print('-------------------------------------------------------------------')
    print('Building Data Set: {}'.format(cls_dataset.data_set_name))
    cls_dataset.build_data_set(input_image_dir=cls_data_dir,
                                          output_dir=data_dir)


    return

def extract_classification_images(data_dir):
    data = input_data.read_data_sets('data/fashion')

    train_images = data.train.images[0:1000]
    train_labels = data.train.labels[0:1000]

    test_images = data.test.images[0:200]
    test_labels = data.test.labels[0:200]

    phases = ['train', 'test']
    root_data_dir = data_dir
    if not os.path.exists(root_data_dir):
        os.mkdir(root_data_dir)

    for phase in phases:
        phase_dir = os.path.join(root_data_dir, phase)
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)
        for label in set(train_labels):
            label_dir = os.path.join(phase_dir, str(label))
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

        for img, label in zip(train_images, train_labels):
            im_count = len(os.listdir(
                os.path.join(root_data_dir, phase, str(label))))
            img_path = os.path.join(root_data_dir,
                                    phase,
                                    str(label),
                                    '{}.bmp'.format(im_count))
            img = img.reshape((28,28))*255
            cv2.imwrite(img_path, img)


    # Clean up
    try:
        shutil.rmtree('data/')
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    print('Extracted: {} images'.format(len(train_images)+len(test_images)))
    print('into -> {}'.format(root_data_dir))
    return root_data_dir
