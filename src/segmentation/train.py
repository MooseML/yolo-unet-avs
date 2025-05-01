import tensorflow as tf
import os
from src.segmentation.utils import iou_metric, dice_score

def compile_model(model, n_classes):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy', iou_metric, dice_score]
    )
    return model



def train_model(model, dataset, epochs=15, buffer_size=500, batch_size=32, deterministic=True):
    """
    Compiles and trains the U-Net model.
    """
    if deterministic:
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()

    dataset = dataset.cache().shuffle(buffer_size)
    history = model.fit(dataset, epochs=epochs)
    return history



def save_model_weights(model, path='outputs/segmentation/unet_weights.h5'):
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    model.save_weights(path)
    print(f"Weights saved to {path}")

def load_model_weights(model, path='outputs/segmentation/unet_weights.h5'):
    if os.path.exists(path):
        model.load_weights(path)
        print(f"Weights loaded from {path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    return model
