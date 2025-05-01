from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2DTranspose, concatenate


def conv_block(inputs, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolutional downsampling block
    """
    conv = Conv2D(n_filters, kernel_size=3, activation='relu',
                  padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, kernel_size=3, activation='relu',
                  padding='same', kernel_initializer='he_normal')(conv)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    next_layer = MaxPooling2D(pool_size=(2, 2))(conv) if max_pooling else conv
    skip_connection = conv
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block

    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Output tensor after upsampling and convolution
    """
    up = Conv2DTranspose(
        n_filters,
        kernel_size=3,
        strides=(2, 2),
        padding='same')(expansive_input)

    merge = concatenate([up, contractive_input], axis=3)

    conv = Conv2D(
        n_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(merge)

    conv = Conv2D(
        n_filters,
        kernel_size=3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal')(conv)

    return conv

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

def unet_model(input_size=(240, 320, 3), n_filters=64, n_classes=23):
    """
    Full U-Net architecture builder.

    Args:
        input_size: Tuple - input image shape (H, W, C)
        n_filters: Int - base number of filters
        n_classes: Int - number of segmentation output classes

    Returns:
        model: tf.keras.Model
    """
    inputs = Input(input_size)

    # Encoder
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters * 2)
    cblock3 = conv_block(cblock2[0], n_filters * 4)
    cblock4 = conv_block(cblock3[0], n_filters * 8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    # Decoder
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, kernel_size=1, padding='same')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    return model
