import matplotlib
matplotlib.use('TkAgg')
#from TFUtils import EvalUtils
#from TFUtils import TrainUtils
#from TFUtils import InOutUtils
from TFUtils import DataSets

def load_data_set_path_dict():
    with open(os.path.join(os.environ['DCNN_DATASETS_PATH'], 'od_ds_paths.json'), 'r') as f:
        ds_path_dict = json.load(f)
    return ds_path_dict

def load_model_path_dict():
    with open(os.path.join(os.environ['TF_OD_MODEL_ROOT'], 'tf_odm_paths.json'), 'r') as f:
        model_path_dict = json.load(f)
    return model_path_dict
