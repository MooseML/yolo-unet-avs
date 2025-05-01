import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from src.detection.yad2k.models.keras_yolo import yolo_head
from src.detection.yad2k.utils.utils import preprocess_image, draw_boxes, get_colors_for_classes
from src.detection.yolo_utils import yolo_eval

def predict_on_image(yolo_model, image_file, anchors, class_names, model_image_size=(608, 608), out_path="out", max_boxes=10, score_threshold=0.3, iou_threshold=0.5):
    """
    Run YOLO object detection on a single image, draw predictions, and display the result

    Args:
    yolo_model (keras.Model): Preloaded YOLO model
    image_file (str): Path to the input image
    anchors (np.ndarray): YOLO anchor boxes
    class_names (list): List of COCO class names
    model_image_size (tuple): Input size expected by the model (width, height)
    out_path (str): Output directory to save annotated image
    max_boxes (int): Max number of boxes after NMS
    score_threshold (float): Minimum score to keep a predicted box
    iou_threshold (float): IoU threshold for non-max suppression

    Returns:
    out_scores (np.ndarray): Confidence scores for predicted boxes
    out_boxes (np.ndarray): Predicted box coordinates
    out_classes (np.ndarray): Class indices for predictions
    output_file (str): Path to saved image with drawn boxes
    """
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image file not found: {image_file}")

    # preprocess image
    image, image_data = preprocess_image(image_file, model_image_size)

    # fwd pass and decode
    yolo_raw_output = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_raw_output, anchors, len(class_names))

    # post-process predictions
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, image_shape=[image.size[1], image.size[0]], max_boxes=max_boxes, score_threshold=score_threshold, iou_threshold=iou_threshold,)

    print(f"Detected {len(out_boxes)} objects in {os.path.basename(image_file)}")

    # draw bounding boxes
    colors = get_colors_for_classes(len(class_names))
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)

    # save
    #  and display
    os.makedirs(out_path, exist_ok=True)
    output_file = os.path.join(out_path, os.path.basename(image_file))
    image.save(output_file)

    plt.figure(figsize=(10, 10))
    plt.imshow(Image.open(output_file))
    plt.axis('off')
    plt.title("Detected Objects")
    plt.show()

    return out_scores, out_boxes, out_classes, output_file


def predict_on_frame(yolo_model, frame, anchors, class_names,
                     model_image_size=(608, 608), max_boxes=10,
                     score_threshold=0.3, iou_threshold=0.5):
    """
    Runs YOLO prediction on a single video frame (NumPy array).
    Returns a frame with boxes drawn.
    """
    # convert OpenCV BGR to PIL RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # preprocess image
    image, image_data = preprocess_image(image, model_image_size)

    # YOLO fwd pass
    yolo_outputs_raw = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_outputs_raw, anchors, len(class_names))

    # eval
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, image_shape=[image.size[1], image.size[0]], max_boxes=max_boxes, score_threshold=score_threshold, iou_threshold=iou_threshold,)

    # draw boxes on image
    colors = get_colors_for_classes(len(class_names))
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)

    # convert back to OpenCV BGR
    output_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return output_frame
