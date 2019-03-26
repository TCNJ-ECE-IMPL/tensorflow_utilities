import os
import argparse
#import tensorflow as tf
from TFUtils.TrainUtils import train_odm, config_odm_run
from TFUtils.InOutUtils import load_data_set_path_dict, load_model_path_dict
from object_detection.utils.config_util import create_pipeline_proto_from_configs, save_pipeline_config

def parse_args():
    parser = argparse.ArgumentParser(description='Script to train a TensorFlow Object Detection Model')

    rgroup = parser.add_argument_group('Required Arguments')

    ds_options = [x for x in os.listdir(os.environ['DCNN_DATASETS_PATH'])]

    #model_choices = load_model_path_dict().keys()

    rgroup.add_argument('--dataset',
                        help='Select ssd_mobilenet_v1_exported_graph set to train model on. Options are listed, to add ssd_mobilenet_v1_exported_graph set follow other instructions',
                        choices=ds_options,
                        required=True,
                        type=str)

    rgroup.add_argument('--pipeline_config_path',
                        help='Path to the config file to use for training',
                        required=True,
                        type=str)

    rgroup.add_argument('--exp_dir',
                        help='Directory to write model ssd_mobilenet_v1_exported_graph to. Name this somethings spoecific to model arch and params',
                        required=True,
                        type=str)

    rgroup.add_argument('--num_train_steps',
                        help='Number of training iterations to train for. One iteration is 1 update of the weights.  10,000 is recommended',
                        required=True,
                        type=int)

    rgroup.add_argument('--num_eval_steps',
                        help='Number of eval iterations to train for. This must set to the number of evaluation images in your data set',
                        required=True,
                        type=int)

    ogroup = parser.add_argument_group('Optional Arguments')

    ogroup.add_argument('--fine_tune_dir',
                        help='Path to directory containing pretrained model checkpoints',
                        default='',
                        required=False,
                        type=str)

    args = parser.parse_args()

    ds_info = load_data_set_path_dict()[args.dataset]
    if ds_info['data_set_type'] != 'object_detection':
        assert('Dataset TypeError: Select a dataset for object detection')
    return args


if __name__ == '__main__':
    args = parse_args()
    # Defining the output_path that the pipeline config will be written to
    pipeline_out_path = os.path.join(args.exp_dir, 'pipeline.config')

    dataset_dir = os.path.join(os.environ['DCNN_DATASETS_PATH'], args.dataset)

    params_proto = config_odm_run(pipline_config_path=args.pipeline_config_path,
                                  dataset_path=dataset_dir,
                                  fine_tune_dir=args.fine_tune_dir)

    save_pipeline_config(params_proto, args.exp_dir)

    print('-'*50)
    print('Beginning Training, logging to {}'.format(args.exp_dir))
    train_odm(model_dir=args.exp_dir,
              pipeline_config_path=pipeline_out_path,
              num_train_steps=args.num_train_steps,
              num_eval_steps=args.num_eval_steps,
              hparams=None)
    print('-'*50)
