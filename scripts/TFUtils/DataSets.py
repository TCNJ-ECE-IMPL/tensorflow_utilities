import os
import cv2
import json
import numpy as np
import tensorflow as tf

# TODO: Improve error handling for loading in a data set

class DataSet:
    """ Class to represent a Data Set created, and annotated by IMPL

    Properties:
        data_set_type:  (str) Data set class (ie. object detection, or basic classification)
        data_set_dir:   (str)

    Methods:
        data_set_description_as_dict():
    """
    def __init__(self, data_set_type=None, data_set_name=None, data_set_description=None, data_set_dir=None):
        self.description_filename = 'data_set_description.json'

        # If the data set exists load the description file
        if data_set_dir:
            self.data_set_dir = data_set_dir
            config_path = os.path.join(self.data_set_dir, 'annotations', self.description_filename)
            self._load_description_file_from_json(config_path)
            return
        if data_set_description:
            self.set_attrs(data_set_description)
            return

        # Setting up basic data set properties
        self.data_set_dir = os.path.join(os.environ['DCNN_DATASETS_PATH'], data_set_name)
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

    def set_attrs(self, dict):
        for key, value in dict.items():
            setattr(self, key, value)
        return

    def _load_description_file_from_json(self, config_path):
        with open(config_path, 'r') as fp:
            config_dict = json.load(fp)
        self.setattrs(config_dict)
        return

    def __repr__(self):
        return 'Data Set Name: {}\nType: {}\nDecription: {}'.format(
            self.data_set_name,
            self.data_set_type,
            self.data_set_description)


class ObjectDetectionDataset(DataSet):
    """ Object Detectiopn Data Set Class to represent the data set (derived from DatSets.DataSet class)

    """
    def __init__(self, data_set_name, data_set_type, data_set_description, num_classes):
        super(ObjectDetectionDataset, self).__init__(data_set_name, data_set_type, data_set_description)
        self.num_classes = num_classes
        self.sub_dirs = []
        return

    def build_data_set(self, output_dir):
        return

class KerasDataset(DataSet):
    def __init__(self, data_set_type=None, data_set_name=None, data_set_description=None, **kwargs):
        data_set_type = 'classification'

        super().__init__(data_set_type=data_set_type,
                         data_set_name=data_set_name,
                         data_set_description=data_set_description,
                         data_set_dir=kwargs.get('data_set_dir'))

        self.train_dir = os.path.join(self.data_set_dir, 'train')
        self.validation_dir = os.path.join(self.data_set_dir, 'validation')

        return

    def get_generators(self, preprocess_input, batch_size):
        self.image_data_gen = self.get_image_generator(preprocess_input)
        return self.train_generator, self.validation_generator

    def get_train_generator(self):
        self.train_generator = image_data_gen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_hgt, self.img_wid),
		    batch_size=32,
		    class_mode='categorical',
            shuffle=True,
            subset='training')
        return train_generator

    def validation_generator(self):
        self.validation_generator = image_data_gen.flow_from_directory(
            self.validation_dir,
            target_size=(IMG_HGT, IMG_WID),
            batch_size=32,
            class_mode='categorical',
            shuffle=False,
            subset='validation')
        return validation_generator

    def get_image_generator(self, preprocess_input, val_split=None):
        image_data_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=val_spit)
        return image_data_gen


class ClassificationDataSet(DataSet):
    """ Data Set Object to represent a Classification Data Set (derived from DataSets.DataSet object)

    Properties:
        data_set_name:          (str)   Data set name (should be unique and descriptive)
        data_set_type:          (str)   Data set type (IMPL tf tools support classification only atm)~~
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

        if not os.path.exists(self.data_set_dir):
            os.mkdir(self.data_set_dir)

        self.image_data_gen = None
        return

    @property
    def image_dir(self):
        return os.path.join(self.data_set_dir, 'images')


    @property
    def train_images(self):
        images, _ = self.load_data_set_from_raw_images(self.image_dir, phase='train')
        return images

    @property
    def train_labels(self):
        _, labels =  self.load_data_set_from_raw_images(self.image_dir, phase='train')
        return labels

    @property
    def validation_images(self):
        images, _ = self.load_data_set_from_raw_images(self.image_dir, phase='validation')
        return images

    @property
    def validation_labels(self):
        _, labels = self.load_data_set_from_raw_images(self.image_dir, phase='validation')
        return labels

    @property
    def train_generator(self):
        self.train_generator = self.image_data_gen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_hgt, self.img_wid),
		    batch_size=32,
		    class_mode='categorical',
            shuffle=True)
        return train_generator

    @property
    def validation_generator(self):
        self.validation_generator = self.image_data_gen.flow_from_directory(
            self.validation_dir,
            target_size=(self.img_hgt, self.img_wid),
            batch_size=32,
            class_mode='categorical',
            shuffle=False)
        return validation_generator

    def get_image_data_generator(self, preprocess_input, val_split=None):
        return ImageDataGenerator(preprocessing_function=preprocess_input)

    def get_generators(self, preprocess_input):
        if not self.image_data_gen:
            self.image_data_gen = get_image_generator(preprocess_input)

        for phase in self.phase_dirs:
            continue
        return self.train_generator, self.validation_generator

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
        # Setting up output dirs
        self.train_dir = os.path.join(self.data_set_dir, 'images/train')
        self.validation_dir = os.path.join(self.data_set_dir, 'images/validation')

        if output_dir:
            output_dir = os.path.join(output_dir, self.data_set_name)
        else:
            output_dir = self.data_set_dir

        if not len(os.listdir(output_dir)):
            self.create_dir_structure(output_dir)

        # Load image file names based on directory structure
        # Loop through phases in 'input_image_dir/' aka train/test/eval
        # TF Record will saved as output_dir/data/phase.record
        phase_dirs = os.listdir(input_image_dir)
        for phase in phase_dirs:
            # Load the images into memory from input dir
            #phase = os.path.basename(phase_dir[:-1] if phase_dir[-1]=='/' else phase_dir)
            phase_dir = os.path.join(self.data_set_dir, 'images/{}'.format(phase))
            if not os.path.exists(phase_dir):
                os.mkdir(phase_dir)

            # Load from input dir
            images, labels = self.load_data_set_from_raw_images(input_image_dir, phase)


            self.label_map = self.get_label_map(set(labels))

            labels = [self.label_map[x] for x in labels]
            self.num_classes = len(set(labels))
            # Save to image dir
            self.write_raw_images(phase_dir, images, labels)

            print('Copied {} train images {} for Dataset usage'.format(len(images), phase.upper()))
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

        print('Successfully built data set assets')
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

    def load_data_set_from_raw_images(self, top_dir, phase):
        """ Function to load images and labels from a directory structure

        Parameters:
            top_dir:    (str) Either 'test' or 'train'

        Returns:
             images:    ([ndarray])
             labels:    ([str])
        """
        if not ((phase == 'test') or (phase == 'train')):
            raise('Argument Error . . . must provide either "test" or "train"')

        image_dir = os.path.join(top_dir, phase)
        image_label_dict = self._get_image_label_dict(image_dir)
        images = []
        labels = []
        for image_path, cls in image_label_dict.items():
            labels.append(cls)
            try:
                img_data = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
                images.append(img_data)
            except:
                raise(IOError, 'Could not read in image: {}'.format(image_path))
        return images, labels

    def get_label_map(self, labels):
        map = {}
        for i, label in enumerate(labels):
            map[label] = str(i)
        return map

    def write_raw_images(self, output_dir, images, labels):
        [os.mkdir(os.path.join(output_dir, label)) for label in set(labels)]

        i = 0
        for image, label in zip(images, labels):

            filename = '{}/{}_{:08}.png'.format(label, label, i)
            outpath = os.path.join(output_dir, filename)
            r = cv2.imwrite(outpath, image, )
            i = i + 1

        return

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
            if not os.path.isdir(os.path.join(top_dir,cls)):
                continue
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

class SegmentationDataSet(DataSet):
    def __init__(self, data_set_type=None, data_set_name=None, data_set_description=None, **kwargs):
        data_set_type = 'segmentation'

        super().__init__(data_set_type=data_set_type,
                         data_set_name=data_set_name,
                         data_set_description=data_set_description,
                         data_set_dir=kwargs.get('data_set_dir'))

        if not os.path.exists(self.data_set_dir):
            os.mkdir(self.data_set_dir)

        self.image_data_gen = None
        self.train_dir = os.path.join(self.data_set_dir, 'images/train')
        self.validation_dir = os.path.join(self.data_set_dir, 'images/test')
        return

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
                files contain records that have features that correspond to "image", and "target"
            - Writes data_set_description.json to output_dir/annotations/ which dumps this object's (SegmentationDataSet)
                properties to a json file for later use
        """
        # Setting up output dirs

        if output_dir:
            output_dir = os.path.join(output_dir, self.data_set_name)
        else:
            output_dir = self.data_set_dir

        if not len(os.listdir(output_dir)):
            self.create_dir_structure(output_dir)

        # Load image file names based on directory structure
        # Loop through phases in 'input_image_dir/' aka train/test/eval
        # TF Record will saved as output_dir/data/phase.record
        phase_dirs = os.listdir(input_image_dir)
        for phase in phase_dirs:
            # Load the images into memory from input dir
            #phase = os.path.basename(phase_dir[:-1] if phase_dir[-1]=='/' else phase_dir)
            phase_dir = os.path.join(self.data_set_dir, 'images/{}'.format(phase))
            if not os.path.exists(phase_dir):
                os.mkdir(phase_dir)

            # Load from input dir
            images, targets = self.load_data_set_from_raw_images(input_image_dir, phase)

            # Save to image dir
            self.write_raw_images(phase_dir, images, targets)

            print('Copied {} train images {} for Dataset usage'.format(len(images), phase.upper()))

        # Write auxiliary files
        ds_description_dict = self.data_set_description_as_dict()
        description_path = os.path.join(output_dir, 'annotations', self.description_filename)
        with open(description_path, 'w') as fp:
            json.dump(ds_description_dict, fp)

        print('Successfully built data set assets')
        return

    def load_data_set_from_raw_images(self, top_dir, phase):
        """ Function to load images and targets from a directory structure

        Parameters:
            top_dir:    (str) Either 'test' or 'train'

        Returns:
             images:    ([ndarray])
             targets:    ([ndarray])
        """
        if not ((phase == 'test') or (phase == 'train')):
            raise('Argument Error . . . must provide either "test" or "train"')

        image_dir = os.path.join(top_dir, phase)
        image_label_dict = self._get_image_label_dict(image_dir)
        images = []
        targets = []
        for image_path, cls in image_label_dict.items():
            try:
                img_data = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
                images.append(img_data)
            except:
                raise(IOError, 'Could not read in image: {}'.format(image_path))
            try:
                target_data = cv2.imread(image_path.split('.') + "_map.bmp", cv2.IMREAD_COLOR).astype(np.float32)
                targets.append(target_data)
            except:
                raise(IOError, 'Could not read in image: {}'.format(image_path.split('.') + "_map.bmp"))
        return images, targets

    def write_raw_images(self, output_dir, images, targets):

        i = 0
        for image, target in zip(images, targets):

            image_filename = '{:08}.png'.format(i)
            target_filename = '{:08}_map.png'.format(i)
            image_outpath = os.path.join(output_dir, image_filename)
            target_outpath = os.path.join(output_dir, target_filename)
            r = cv2.imwrite(outpath, image, )
            r = cv2.imwrite(outpath, target, )
            i = i + 1

        return
