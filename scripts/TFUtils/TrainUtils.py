import os
import tensorflow as tf
from .InOutUtils import load_data_set_path_dict, load_model_path_dict
from object_detection import model_lib
from object_detection import model_hparams
from object_detection.utils.config_util import get_configs_from_pipeline_file, create_pipeline_proto_from_configs


def train_odm(model_dir, pipeline_config_path, num_train_steps, num_eval_steps, hparams):
    """ Function to execute the training and evaluation of a TensorFlow Object Detection Moodel on
        a specified ssd_mobilenet_v1_exported_graph set.

    Parameters:
        model_dir: (str)
        pipeline_config_path: (str)
        num_train_steps: (int)
        num_eval_steps: (int)
        hparams: (str) Contains ssd_mobilenet_v1_exported_graph set parameters

    Returns:
        0: On completed execution of num_train_steps along with num_eval_steps of evaluations
    """
    # Generate the estimator and set the model dir of the estimator to the exp dir
    #   This dir will contain the model weights and training stats
    config = tf.estimator.RunConfig(model_dir=model_dir)
    print(config)
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(hparams),
        pipeline_config_path=pipeline_config_path,
        train_steps=num_train_steps,
        eval_steps=num_eval_steps)
    # Parse out the needed items from the estimator
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fn = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    #eval_steps = train_and_eval_dict['eval_steps']

    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fn,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
    return 0


def config_odm_run(pipline_config_path, dataset_path, fine_tune_dir=None):
    """ Function to modify the pipeline config. Modifies the file for a specific training data set and model fine tune ckpt

    Parameters
        pipeline_config_path
        data_set_path
        fine_tune_dir

    Returns:
         pipeline_proto: (??) Protobuf to use when training object detection model
    """

	#TODO: Set num classes from inside this function, right now provided config file must have it set
	#	Maybe we need a field in the data set description json for num_classes

    pipeline_dict = get_configs_from_pipeline_file(pipline_config_path)
    pipeline_dict['train_input_config'].label_map_path = os.path.join(dataset_path, 'annotations', 'label_map.pbtxt')
    pipeline_dict['train_input_config'].tf_record_input_reader.input_path[0] = os.path.join(dataset_path, 'data', 'train.record')
    pipeline_dict['eval_input_config'].label_map_path = os.path.join(dataset_path, 'annotations', 'label_map.pbtxt')
    pipeline_dict['eval_input_config'].tf_record_input_reader.input_path[0] = os.path.join(dataset_path, 'data',  'eval.record')
    pipeline_dict['train_config'].fine_tune_checkpoint = ''

    if fine_tune_dir:
        pipeline_dict['train_config'].fine_tune_checkpoint = os.path.join(fine_tune_path, 'model.ckpt')

    pipeline_proto = create_pipeline_proto_from_configs(pipeline_dict)
    return pipeline_proto
