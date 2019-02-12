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
                                data_set_name='Fashion-MNIST',
                                data_set_description='Dataset created during testing of data set')
    cls_dataset.create_dir_structure(os.path.join(data_dir, cls_dataset.data_set_name))
    cls_data_dir = extract_classification_images(cls_dataset.data_set_dir+'/images/')
    print('-------------------------------------------------------------------')
    print('Building Data Set: {}'.format(cls_dataset.data_set_name))
    cls_dataset.build_data_set(input_image_dir=cls_data_dir,
                                          output_dir=data_dir)
    print('Everything OK so far ...')

    return

def extract_classification_images(data_dir):
    download_dir = os.path.join(data_dir,'download')
    data = input_data.read_data_sets(download_dir)

    phases = ['train', 'test']
    data_dict = {'train': {}, 'test': {}}

    data_dict['train']['images'] = data.train.images[0:1000]
    data_dict['train']['labels'] = data.train.labels[0:1000]

    data_dict['test']['images'] = data.test.images[0:200]
    data_dict['test']['labels'] = data.test.labels[0:200]

    root_data_dir = data_dir
    if not os.path.exists(root_data_dir):
        os.mkdir(root_data_dir)

    for phase in phases:
        phase_dir = os.path.join(root_data_dir, phase)
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)
        for label in set(data_dict[phase]['labels']):
            label_dir = os.path.join(phase_dir, str(label))
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

        images = data_dict[phase]['images']
        labels = data_dict[phase]['labels']
        for img, label in zip(images, labels):
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
        shutil.rmtree(download_dir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    print('Extracted: {} images'.format(
        len(data_dict['train']['images']) + len(data_dict['test']['images'])))
    print('into -> {}'.format(root_data_dir))
    return root_data_dir
