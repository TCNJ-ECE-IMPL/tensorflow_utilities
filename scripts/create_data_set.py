import argparse
from TFUtils.DataSets import ClassificationDataSet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to create a new data set registry')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--data_set_type',
                        choices=['object_detection', 'classification'],
                        help='Data set type describes the format this script will save the TensorFlow data',
                        required=True,
                        type=str)

    rgroup.add_argument('--data_set_name',
                        help='Choose a name for the data set. The data will be saved to [output_dir]/[data_set_name]',
                        required=True,
                        type=str)

    rgroup.add_argument('--input_image_dir',
                        help='Directory path where the data files exist, must be in the following format\n'\
                                'input_image_dir/\n'\
                                '   phase1/\n'\
                                '       class1/\n'\
                                '           img1.jpg\n'\
                                '       class2/\n'\
                                '           dog.jpg\n'\
                                '   phase2/\n'\
                                '       class1/\n'\
                                '           cat.png\n'\
                                '       class2/\n'\
                                '           doggo.bmp\n'\
                                'Where phase1, phase2, etc are train, val, test (dont need all three) and class ids are\n'\
                                'subdirectories names.\n',
                        required=True,
                        type=str)

    ogroup = parser.add_argument_group('Optional Arguments')

    ogroup.add_argument('--output_dir',
                        help='Data will be written to this directory',
                        required=False,
                        type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    description = input('Enter a description for the data set:\n')

    if args.data_set_type == 'classification':
        classDS = ClassificationDataSet(data_set_name=args.data_set_name,
                                        data_set_type=args.data_set_type,
                                        data_set_description=description)

        classDS.build_data_set(input_image_dir=args.input_image_dir,
                               output_dir=args.output_dir)
