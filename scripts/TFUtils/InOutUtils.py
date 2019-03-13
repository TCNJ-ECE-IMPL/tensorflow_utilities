import os
import json

def load_data_set_path_dict():
    data_set_path = os.environ['DCNN_DATASETS_PATH']
    data_set_dict = {}
    for data_set in os.listdir(data_set_path):
        data_set_desc = '{}/{}/annotations/data_set_description.json'.format(data_set_path, data_set)
        with open(data_set_desc, 'r') as f:
            data_set_dict[data_set] = json.load(f)

    return data_set_dict

def load_model_path_dict():
    with open(os.path.join(os.environ['TF_OD_MODEL_ROOT'], 'tf_odm_paths.json'), 'r') as f:
        model_path_dict = json.load(f)
    return model_path_dict
