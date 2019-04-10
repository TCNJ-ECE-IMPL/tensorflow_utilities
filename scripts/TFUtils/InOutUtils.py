import os
import json

def load_data_set_info_from_json(data_set_id):
    data_set_path = os.environ['DCNN_DATASETS_PATH']
    data_set_dict = {}
    for data_set in os.listdir(data_set_path):
        if not os.path.isdir(os.path.join(data_set_path,data_set)):
            continue
        data_set_desc = '{}/{}/annotations/data_set_description.json'.format(data_set_path, data_set)
        with open(data_set_desc, 'r') as f:
            data_set_dict[data_set] = json.load(f)
    return data_set_dict[data_set_id]

def load_data_set_info_from_dir(top_dir):
    def _get_num_images(top_dir):
        num_images = 0
        for root, directories, filenames in os.walk(top_dir):

            for directory in directories:
                dir = os.path.join(root, directory)
                if os.path.isdir(dir):
                    print(root, directory)
                    num_images += len(os.listdir(os.path.join(root, directory)))

        return num_images

    def _get_label_map(labels):
        map = {}
        for i, label in enumerate(labels):
            map[label] = i

        return map

    train_dir = os.path.join(top_dir, 'train')
    val_dir = os.path.join(top_dir, 'validation')
    train_labels = [x for x in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, x))]
    val_labels = [x for x in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, x))]
    assert(train_labels == val_labels)
    labels = train_labels

    dataset_size = {'train': _get_num_images(train_dir),
                    'validation': _get_num_images(val_dir)}

    #print(dataset_size)
    #print(labels)

    # phases = os.listdir(top_dir)
    # labels = []
    # data_set_size = {}
    # for phase  in phases:
    #     path = os.path.join(top_dir, phase)
    #     labels = os.listdir(path)
    #
    #     data_set_size = os.path.listdir(os.path.join(path, ))

    data_set_dict = {
        'data_set_name': top_dir,
        'data_set_dir': top_dir,
        'train_dir': train_dir,
        'validation_dir': val_dir,
        'label_map': _get_label_map(labels),
        'classes': list(_get_label_map(labels).values()),
        'data_set_type': 'classification',
        'data_set_size': dataset_size,
        'data_set_description': "Test"
    }


    return data_set_dict
