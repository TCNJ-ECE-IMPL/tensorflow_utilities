import os
import shutil
import argparse
from tests import *

tmp_dir = 'scripts/tests/tmp'

def parse_args():
    parser = argparse.ArgumentParser(description='Script to test repository functionality')

    ogroup = parser.add_argument_group('Optional Arguments')

    ogroup.add_argument('--all',
                        help='Runs all tests',
                        required=False,
                        action='store_true')

    ogroup.add_argument('--save',
                        help='If FLAG is included script will keep generated data set',
                        required=False,
                        action='store_true')

    ogroup.add_argument('--data_dir',
                        help='Directory to save data set to',
                        required=False,
                        type=str,
                        default=tmp_dir)

    args = parser.parse_args()

    return args

def main(args):

    if args.all:

        if not args.save:
            dir = args.data_dir
        else:
            dir = os.environ['DCNN_DATASETS_PATH']
        if not os.path.exists(dir):
            os.mkdir(dir)

        test_data_set(dir)

        if not args.save:
            shutil.rmtree(dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
