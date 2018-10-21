import argparse
import tensorflow as tf
from TFUtils.DataSets import ClassificationDataSet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to create a TF Record File')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--data_set_type',
                        choices=['object_detection', 'classification'],
                        help='Choose data set type based off of end model goal. See data set spec in TeamDrive/infrastructure for more help',
                        required=True,
                        type=str)

    rgroup.add_argument('--input_image_dir',
                        help='Directory path containing the top directory of the data set (format specified by TeamDrive/infrastructure data set spec',
                        required=True,
                        type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.data_set_type == 'classification':
        classDS = ClassificationDataSet(data_set_name='test1',
                                        data_set_type=args.data_set_type,
                                        data_set_description='Description')
        classDS.build_data_set(input_image_dir=args.input_image_dir)