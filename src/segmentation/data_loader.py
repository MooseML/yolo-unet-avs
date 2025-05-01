# src/segmentation/data_loader.py

import os
import tensorflow as tf

def load_image_and_mask_paths(image_dir: str, mask_dir: str):
    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))  
    image_paths = [os.path.join(image_dir, f) for f in image_filenames]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_filenames]

    return image_paths, mask_paths

def preview_image_pair(image_path, mask_path):
    """
    Visualize one image/mask pair using imageio and matplotlib.
    For notebook use
    """
    import imageio
    import matplotlib.pyplot as plt

    img = imageio.v2.imread(image_path)
    mask = imageio.v2.imread(mask_path)

    fig, arr = plt.subplots(1, 2, figsize=(14, 10))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[1].imshow(mask[:, :, 0])
    arr[1].set_title('Segmentation')
    plt.show()

def decode_image_and_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)  # or channels=1 if masks are grayscale
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return image, mask

def resize_image_and_mask(image, mask, image_size=(240, 1320)):
    image = tf.image.resize(image, image_size, method='nearest')
    mask = tf.image.resize(mask, image_size, method='nearest')
    return image, mask

def load_dataset(image_paths, mask_paths, image_size=(240, 320), shuffle=True, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    dataset = dataset.map(decode_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(lambda x, y: resize_image_and_mask(x, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

