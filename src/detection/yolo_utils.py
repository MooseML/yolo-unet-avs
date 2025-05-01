
import tensorflow as tf 
from src.detection.yad2k.models.keras_yolo import yolo_head
from src.detection.yad2k.utils.utils import (draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image)

import tensorflow as tf

def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    """
    Filters YOLO predicted boxes based on objectness score and class probability

    Args:
    boxes (tf.Tensor): Shape (grid_h, grid_w, num_anchors, 4) — predicted box coords (bx, by, bh, bw)
    box_confidence (tf.Tensor): Shape (grid_h, grid_w, num_anchors, 1) — objectness score per box
    box_class_probs (tf.Tensor): Shape (grid_h, grid_w, num_anchors, num_classes) — class probabilities
    threshold (float): Minimum confidence score to retain a prediction

    Returns:
    scores (tf.Tensor): Shape (None,) — retained class probability scores
    boxes (tf.Tensor): Shape (None, 4) — filtered bounding box coordinates
    classes (tf.Tensor): Shape (None,) — predicted class indices for each box
    """
    box_scores = box_confidence * box_class_probs # (grid, grid, anchors, classes)
    box_classes = tf.math.argmax(box_scores, axis=-1) # most likely class per box
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1) # associated max score per box

    # filter boxes by confidence threshold
    mask = box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores, mask)
    boxes = tf.boolean_mask(boxes, mask)
    classes = tf.boolean_mask(box_classes, mask)

    return scores, boxes, classes


def iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) btw two bounding boxes

    Args:
    box1 (list or tuple): [x1, y1, x2, y2] for the first box
    box2 (list or tuple): [x1, y1, x2, y2] for the second box

    Returns:
    float: IoU between the two boxes (range 0 to 1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(xi2 - xi1, 0)
    inter_height = max(yi2 - yi1, 0)
    inter_area = inter_width * inter_height

    # areas of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    # avoid div by zero
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to remove overlapping predictions

    Args:
    scores (tf.Tensor): Shape (None,) — class probability scores
    boxes (tf.Tensor): Shape (None, 4) — (y_min, x_min, y_max, x_max) coords scaled to image size
    classes (tf.Tensor): Shape (None,) — predicted class indices
    max_boxes (int): Max number of boxes to keep
    iou_threshold (float): IoU threshold for NMS

    Returns:
    scores, boxes, classes: Tensors filtered by NMS, all ≤ max_boxes in length
    """
    boxes = tf.cast(boxes, tf.float32)
    scores = tf.cast(scores, tf.float32)

    selected_indices_all = []

    for cls in tf.unique(classes)[0]:
        mask = (classes == cls)
        cls_boxes = tf.boolean_mask(boxes, mask)
        cls_scores = tf.boolean_mask(scores, mask)
        cls_indices = tf.squeeze(tf.where(mask), axis=1)

        if tf.shape(cls_scores)[0] > 0:
            nms = tf.image.non_max_suppression(cls_boxes, cls_scores, max_output_size=max_boxes, iou_threshold=iou_threshold)
            selected = tf.gather(cls_indices, nms)
            selected_indices_all.append(selected)

    if not selected_indices_all:
        return tf.constant([]), tf.constant([]), tf.constant([])

    final_indices = tf.concat(selected_indices_all, axis=0)

    # sort final detections by score
    final_scores = tf.gather(scores, final_indices)
    top_k = tf.argsort(final_scores, direction="DESCENDING")[:max_boxes]

    scores = tf.gather(final_scores, top_k)
    boxes = tf.gather(boxes, tf.gather(final_indices, top_k))
    classes = tf.gather(classes, tf.gather(final_indices, top_k))

    return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """
    Converts YOLO box format (center x/y + width/height) to corners (y_min, x_min, y_max, x_max)

    Args:
    box_xy (tf.Tensor): Shape (..., 2) — center x and y
    box_wh (tf.Tensor): Shape (..., 2) — width and height

    Returns:
    tf.Tensor: Shape (..., 4) corners in order: y_min, x_min, y_max, x_max
    """
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.concat([
        box_mins[..., 1:2], # y_min
        box_mins[..., 0:1], # x_min
        box_maxes[..., 1:2], # y_max
        box_maxes[..., 0:1]  # x_max
    ], axis=-1)



def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    Post-processes YOLO model outputs: filters, rescales, and applies NMS

    Args:
    yolo_outputs (tuple): Output tensors from yolo_head: box_xy, box_wh, box_confidence, box_class_probs
    image_shape (tuple): Original image shape (height, width)
    max_boxes (int): Maximum number of boxes to keep
    score_threshold (float): Confidence threshold
    iou_threshold (float): IoU threshold for NMS

    Returns:
    scores, boxes, classes: Filtered and rescaled predictions
    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)
    boxes = scale_boxes(boxes, image_shape)

    return yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes, iou_threshold=iou_threshold)

