import tensorflow as tf
from utils import label_map_util

def load_frozen_graph(frozen_graph_path):
    """ Function to load the saved model parameters for inferencing

    Parameters:
        frozen_graph_path: Path to the frozen inference graph saved during training

    Returns:
        detection_graph: Tensorflow Graph of the loaded Object Detection Model
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def label_map_to_cat_idx(label_map_path, num_classes):
    """ This function produces a category index from a label map file path and a number of classes.
            The category index is a dictionary that maps class ids to class names

    Parameters:
        label_map_path: Path to the label map used to train the object detection model (str)
        num_classes: Number of classes the model was trained on (int)

    Returns:
        category index: dict mapping class ids (int) to class names (str)
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index
