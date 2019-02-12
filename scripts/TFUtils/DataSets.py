import os
import cv2
import json
import numpy as np
import tensorflow as tf

# TODO: Extract some functionality out of children classes into parent

def load_data_set_path_dict():
    with open(os.path.join(os.environ['DCNN_DATASETS_PATH'], 'od_ds_paths.json'), 'r') as f:
        ds_path_dict = json.load(f)
    return ds_path_dict

def load_model_path_dict():
    with open(os.path.join(os.environ['TF_OD_MODEL_ROOT'], 'tf_odm_paths.json'), 'r') as f:
        model_path_dict = json.load(f)
    return model_path_dict

class DataSet:
    """ Class to represent a Data Set created, and annotated by IMPL (ie. racoon_data_set)

    Properties:
        data_set_type:  (str) Data set class (ie. object detection, or basic classification)
        data_set_dir:   (str)

    Methods:
        data_set_description_as_dict():
    """
    def __init__(self, data_set_type, data_set_name, data_set_description, data_set_dir=None):
        self.description_filename = 'data_set_description.json'
        if data_set_dir:
            self.data_set_dir = data_set_dir
            config_path = os.path.join(self.data_set_dir, 'annotations', self.description_filename)
            self._load_description_file_from_json(config_path)
            return
        # Setting up basic data set properties
        self.data_set_type = data_set_type
        self.data_set_name = data_set_name
        self.data_set_description = data_set_description

        self.sub_dirs = ['annotations', 'data', 'images']
        self.data_set_size = {}
        return

    def data_set_description_as_dict(self):
        prop_dict = {}
        for property, value in vars(self).items():
            prop_dict[property] = value
        return prop_dict

    def load_data_set(self):
        AssertionError('Abstract method, implement in children')
        return

    def _load_description_file_from_json(self, config_path):
        with open(config_path, 'r') as fp:
            config_dict = json.load(fp)
        for key, value in config_dict.items():
            setattr(self, key, value)
        return
    def __repr__(self):
        print('Data Set Name: {}\nType: {}\nDecription: {}'.format(
            self.data_set_name,
            self.data_set_type,
            self.data_set_description))


class ObjectDetectionDataSet(DataSet):
    """ Object Detectiopn Data Set Class to represent the data set (derived from DatSets.DataSet class)

    """
    def __init__(self, data_set_name, data_set_type, data_set_description, num_classes):
        super(Dataset, self).__init__(data_set_name, data_set_type, data_set_description)
        self.num_classes = num_classes
        self.sub_dirs = []
        return

    def build_data_set(self, output_dir):
        return

class ClassificationDataSet(DataSet):
    """ Data Set Object to represent a Classification Data Set (derived from DataSets.DataSet object)

    Properties:
        data_set_name:          (str)   Data set name (should be unique and descriptive)
        ~~data_set_type:          (str)   Data set type (IMPL tf tools support classification only atm)~~
        data_set_description:   (str)   Data set description can be provided to give more detail on the data
        classes:                (list)  List of unique class ids gathered from data set creation
        features:               (dict)  Dict of feature details to use when extracting record from the TFRecord
    Methods:
        build_data_set():
        load_data_set_from_TFRecordIO():
        load_data_set_from_csv():
        load_data_set_from_raw_images():
        create_data_set_from_raw_images():
        create_data_set_from_csv():
    """
    def __init__(self, data_set_type=None, data_set_name=None, data_set_description=None, **kwargs):
        self.classes = []
        self.features = {}
        data_set_type = 'classification'

        super().__init__(data_set_type=data_set_type,
                         data_set_name=data_set_name,
                         data_set_description=data_set_description,
                         data_set_dir=kwargs.get('data_set_dir'))

        return

    def load_data_set_as_tensors(self):
        try:
            data_dict = self.load_data_set_from_TFRecordIO()
        except:
            try:
                data_dict = self.load_data_set_from_images()
            except:
                try:
                    data_dict = self.load_data_set_from_csv()
                except:
                    AssertionError('Data Set has no image or label data...\nCreate a data set using create_data_set.py')

        return data_dict

    def build_data_set(self, input_image_dir, output_dir=None):
        """ Function to build a data set and format it so that way this class can read its descriptions, images, and
            tf record files

        Parameters:
            input_image_dir:    (str) Directory path where the data files exist, must be in the following format
                                input_image_dir/
                                    phase1/
                                        class1/
                                            img1.jpg
                                        class2/
                                            dog.jpg
                                    phase2/
                                        class1/
                                            cat.png
                                        class2/
                                            doggo.bmp

                                Where phase1, phase2, etc are train, val, test (dont need all three) and class ids are
                                subdirectories names.
            output_dir:         (str) [Default=None] This causes the data set to be created and registered in the
                                $OD_DS_ROOT directory

        Returns:
            This doesnt return anything but does have side effects,

            - Creates a directory structure
                output_dir/
                    annotations/
                    data/
                    images/
            - Writes phaseN.record to output_dir/data/phaseN.record for every phase in the input_image_dir. The .record
                files contain records that have features that correspond to "image", and "label"
            - Writes data_set_description.json to output_dir/annotations/ which dumps this object's (ClassificationDataSet)
                properties to a json file for later use
        """
        
        assert os.path.exists(input_image_dir)
        if output_dir:
            output_dir = os.path.join(output_dir, self.data_set_name)
        else:
            # Setting up output dirs
            output_dir = os.path.join(os.environ['DCNN_DATASETS_PATH'], self.data_set_name)


        if len(os.listdir(output_dir)):
            self.create_dir_structure(output_dir)
        # Load image file names based on directory structure
        # Loop through phases in 'input_image_dir/' aka train/test/eval
        # TF Record will saved as output_dir/data/phase.record
        phase_dirs = os.listdir(input_image_dir)
        for phase_dir in phase_dirs:
            # Load the images into memory in TFRecord format
            phase = os.path.basename(phase_dir[:-1] if phase_dir[-1]=='/' else phase_dir)
            images, labels = self.load_data_set_from_raw_images(os.path.join(input_image_dir, phase_dir))

            # Define the data sets classes and assert an error if a phase a different number of
            # classes that previously assumed
            if self.classes==[]:
                self.classes = list(set(labels))
            else:
                assert set(labels) == set(self.classes)
            self.data_set_size[phase] = len(images)

            # Create features to keep track of in data set description files
            # These can help users of the data set figure out how to extract the TFRecord files
            self.features = {'image': {'name': 'image', 'type': 'bytes'},
                            'label': {'name': 'label', 'type': 'bytes'}}

            # Create TF record
            output_path = os.path.join(output_dir, 'data', '{}.record'.format(phase))
            self.write_record(output_path, images, labels, self.features)
            print('Succesfully Wrote TF Record for {} image set\nCount: {}'.format(phase.upper(), self.data_set_size[phase]))

        # Write auxiliary files
        ds_description_dict = self.data_set_description_as_dict()
        description_path = os.path.join(output_dir, 'annotations', self.description_filename)
        with open(description_path, 'w') as fp:
            json.dump(ds_description_dict, fp)
        return

    def load_data_set_from_TFRecordIO(self):
        """ Not Implemented Yet """
        top_dir = os.path.join(self.data_set_dir, 'data')
        result= {}
        for recIO in os.listdir(top_dir):
            recIO_path = os.path.join(top_dir, recIO)
            # phase is 'phase.record'.split('.')[0] => 'phase' aka train.rec=>'train'
            phase = os.path.basename(recIO.split('.')[0])

            with tf.Session() as sess:
                result[phase] = {}
                feature = {self.features['image']['name']: tf.FixedLenFeature([], tf.string),
                           self.features['label']['name']: tf.FixedLenFeature([], tf.string)}

                # Create a list of filenames and pass it to a queue
                filename_queue = tf.train.string_input_producer([recIO_path], num_epochs=1)
                # Define a reader and read the next record
                reader = tf.TFRecordReader()
                _, serialized_example = reader.read(filename_queue)
                # Decode the record read by the reader
                dec_features = tf.parse_single_example(serialized_example, features=feature)
                # Convert the image data from string back to the numbers
                image = tf.decode_raw(dec_features[self.features['image']['name']], tf.float32)

                # Cast label data into int32
                label = tf.cast(dec_features[self.features['label']['name']], tf.int32)
                # Reshape image data into the original shape
                image = tf.reshape(image, [224, 224, 3])

                # Any preprocessing here ...

                # Creates batches by randomly shuffling tensors
                images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
                                                        min_after_dequeue=10)
                result[phase][self.features['image']['name']] = images
                result[phase][self.features['label']['name']] = labels
        return result

    def load_data_set_from_csv(self, csv_filepath):
        """ Not Implemented Yet """
        self._csv_to_image_label_dict(csv_filepath)
        return

    def load_data_set_from_raw_images(self, top_dir):
        """ Function to load images and labels from a directory structure

        Parameters:
            top_dir:    (str)

        Returns:
             images:    ([ndarray])
             labels:    ([str])
        """
        image_label_dict = self._get_image_label_dict(top_dir)
        images = []
        labels = []
        for image_path, cls in image_label_dict.items():
            labels.append(cls)
            try:
                img_data = cv2.imread(image_path).astype(np.float32)
                images.append(img_data)
            except:
                raise(IOError, 'Could not read in image: {}'.format(image_path))
        return images, labels

    def write_record(self, output_path, images, labels, features):
        """ Function to Write a set of images and labels to a TF Record

        Parameters:
            output_path:    (str)
            images:         ([ndarray])
            labels:         ([str])
            features:       (dict)

        Returns:

        """
        writer = tf.python_io.TFRecordWriter(output_path)
        for image, label in zip(images, labels):
            # Create a feature
            feature = {self.features['label']['name']: self._bytes_feature(tf.compat.as_bytes(label)),
                       self.features['image']['name']: self._bytes_feature(tf.compat.as_bytes(image.tostring()))}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        return

    def create_data_set_from_csv(self):
        """ Not Implemented Yet """
        return

    def _get_image_label_dict(self, top_dir):
        """ Helper Function to return a dictionary mapping image paths to class labels"""
        image_label_dict = {}
        classes = os.listdir(top_dir)
        for cls in classes:
            images_list = os.listdir(os.path.join(top_dir, cls))
            for image_fname in images_list:
                image_path = os.path.join(top_dir, cls, image_fname)
                image_label_dict[image_path] = cls
        return image_label_dict

    def _csv_to_image_label_dict(self, csv_filepath):
        """ Not Implemented Yet """
        return

    def _bytes_feature(self, value):
        """ Helper Function to return an object that can be fed to the TF Record """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def create_dir_structure(self, output_dir):
        """ Helper function to create the directory structure needed for data set storage """
        self.data_set_dir = output_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for sub_dir in self.sub_dirs:
            w_dir = os.path.join(output_dir, sub_dir)
            if not os.path.exists(w_dir):
                os.mkdir(w_dir)
        assert os.path.exists(output_dir)
        return 'Successfully Created Directory Structure for {}'.format(self.data_set_name)

    def data_set_description_as_dict(self):
        return super().data_set_description_as_dict()
