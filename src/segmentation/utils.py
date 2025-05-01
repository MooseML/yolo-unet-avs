import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np 
import tensorflow as tf

# def display(display_list, title_list=None):
#     plt.figure(figsize=(15, 15))
#     default_titles = ['Input Image', 'True Mask', 'Predicted Mask']

#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i + 1)
#         plt.title(title_list[i] if title_list else default_titles[i])

#         img = display_list[i]

#         if img.shape[-1] == 1:  # mask (single-channel)
#             plt.imshow(tf.squeeze(img), cmap='nipy_spectral', vmin=0, vmax=22)
#         else:  # image (RGB)
#             plt.imshow(tf.keras.preprocessing.image.array_to_img(img))

#         plt.axis('off')

#     plt.show()

def label_to_rgb(mask, colormap='nipy_spectral', num_classes=23):
    """
    Convert a single-channel mask (class indices) to an RGB image using a fixed colormap.
    """
    if isinstance(mask, tf.Tensor):
        mask = tf.squeeze(mask).numpy()
    else:
        mask = np.squeeze(mask)

    cmap = plt.get_cmap(colormap)
    norm = colors.Normalize(vmin=0, vmax=num_classes - 1)
    rgb_mask = cmap(norm(mask))[..., :3]  # Drop alpha channel

    return (rgb_mask * 255).astype(np.uint8)

def display(display_list, title_list=None):
    """
    Display input image, true mask, and predicted mask side-by-side using consistent colormap.
    """
    plt.figure(figsize=(15, 15))
    default_titles = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title_list[i] if title_list else default_titles[i])

        img = display_list[i]

        if img.shape[-1] == 1:  # mask
            rgb_mask = label_to_rgb(img)
            plt.imshow(rgb_mask)
        else:  # RGB image
            plt.imshow(tf.keras.preprocessing.image.array_to_img(img))

        plt.axis('off')

    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(model, dataset=None, num=1, sample_image=None, sample_mask=None):
    """
    Displays predictions either from a dataset or a single image + mask pair.
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    elif sample_image is not None and sample_mask is not None:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])
    else:
        raise ValueError("Provide either a dataset or a sample_image and sample_mask.")


def iou_metric(y_true, y_pred, num_classes=23):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)

    ious = []
    for i in range(num_classes):
        y_true_class = tf.equal(y_true, i)
        y_pred_class = tf.equal(y_pred, i)
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_true_class, y_pred_class), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(y_true_class, y_pred_class), tf.float32))
        iou = tf.math.divide_no_nan(intersection, union)
        ious.append(iou)

    return tf.reduce_mean(ious)


def dice_score(y_true, y_pred, smooth=1e-6):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.float32)

    intersection = tf.reduce_sum(y_pred * y_true)
    union = tf.reduce_sum(y_pred) + tf.reduce_sum(y_true)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice
