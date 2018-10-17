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
    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(hparams),
        pipeline_config_path=pipeline_config_path,
        train_steps=num_train_steps,
        eval_steps=num_eval_steps)
    # Parse out the needed items from the estimator
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fn = train_and_eval_dict['eval_input_fn']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']
    eval_steps = train_and_eval_dict['eval_steps']
    # Create the train and eval specs from the ssd_mobilenet_v1_exported_graph and configs
    #   This defines what the esimator will
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fn,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_steps,
        eval_on_train_data=False)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
    return 0


def config_odm_run(ds_name, model_name, pipline_config_path):
    """ Function to modify the pipeline config. Modifies the file for a specific training data set and model fine tune ckpt

    Parameters
        ds_name: (str) Name of model to find paths for hparams override
        model_name: (str) Name of the model to train.
                    used to index '$TF_OD_MODELS:/tf_odm_pre_trained_data.json' as a dict

    Returns:
         hparams: (dict) Dict containing config params for training with modified input data set
    """
	#TODO: Set num classes from inside this function, right now provided config file must have it set
	#	Maybe we need a field in the data set description json for num_classes
    ds_path_dict = load_data_set_path_dict()
    model_path_dict = load_model_path_dict()
    ds_root_path = ds_path_dict[ds_name]
    tf_model_path = model_path_dict[model_name]

    pipeline_dict = get_configs_from_pipeline_file(pipline_config_path)
    pipeline_dict['train_input_config'].label_map_path = os.path.join(ds_root_path, 'annotations', 'label_map.pbtxt')
    pipeline_dict['train_input_config'].tf_record_input_reader.input_path[0] = os.path.join(ds_root_path, 'data', 'train.record')
    pipeline_dict['eval_input_config'].label_map_path = os.path.join(ds_root_path, 'annotations', 'label_map.pbtxt')
    pipeline_dict['eval_input_config'].tf_record_input_reader.input_path[0] = os.path.join(ds_root_path, 'data',  'eval.record')
    pipeline_dict['train_config'].fine_tune_checkpoint = os.path.join(tf_model_path, 'model.ckpt')

    pipeline_proto = create_pipeline_proto_from_configs(pipeline_dict)
    return pipeline_proto
