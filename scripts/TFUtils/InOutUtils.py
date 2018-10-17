import os
import json

def load_data_set_path_dict():
    with open(os.path.join(os.environ['OD_DS_ROOT'], 'od_ds_paths.json'), 'r') as f:
        ds_path_dict = json.load(f)
    return ds_path_dict

def load_model_path_dict():
    with open(os.path.join(os.environ['TF_OD_MODEL_ROOT'], 'tf_odm_paths.json'), 'r') as f:
        model_path_dict = json.load(f)
    return model_path_dict