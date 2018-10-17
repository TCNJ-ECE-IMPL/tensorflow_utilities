import cv2
import argparse
import numpy as np
import tensorflow as tf
from TFUtils.EvalUtils import *
from utils import visualization_utils as vis_util


def parse_args():
    parser = argparse.ArgumentParser(description='A script to inference one image on an object detection model')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--frozen_graph',
                        help='Path to frozen inference graph of model to be inferenced',
                        required=True,
                        type=str)

    rgroup.add_argument('--image_path',
                        help='Path to image to be tested',
                        required=True,
                        type=str)

    rgroup.add_argument('--label_map_path',
                        help='Path to label map used for ssd_mobilenet_v1_exported_graph set creation',
                        required=True,
                        type=str)

    rgroup.add_argument('--num_classes',
                        help='Number of classes the model was trained to classify',
                        required=True,
                        type=int)

    return parser.parse_args()


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


if __name__ == '__main__':
    args = parse_args()
    graph_path = args.frozen_graph
    image_path = args.image_path
    label_map_path = args.label_map_path
    num_classes = args.num_classes

    # Load frozen graph
    detection_graph = load_frozen_graph(graph_path)

    # Load label map and convert to category index(this is the format they specify they need it in in order to visualuze
    #  results
    cat_idx = label_map_to_cat_idx(label_map_path, num_classes)

    image = cv2.imread(image_path)
    im_tensor = np.expand_dims(image, axis=0)

    output_dict = run_inference_for_single_image(im_tensor, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        cat_idx,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)

    cv2.imwrite('img.jpg', image)
    print('Done')
