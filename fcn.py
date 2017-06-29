from functools import partial

import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import (
    Input, Conv2D, Conv2DTranspose, MaxPooling2D,
    ZeroPadding2D, Dropout, Cropping2D, merge
)
from keras.utils.data_utils import get_file

import config as cfg

VGG_WEIGHTS_NO_TOP_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'  # noqa


def corssentropy2d(y_true, y_pred):
    num_class = K.int_shape(y_pred)[-1]
    _y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), num_class)
    _y_pred = K.reshape(y_pred, (-1, num_class))
    log_softmax = tf.nn.log_softmax(_y_pred)

    tf.summary.image('true_layout', y_true, max_outputs=8)
    tf.summary.image('pred_layout',
                     tf.to_float(
                         tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=-1)),
                     max_outputs=8)

    l1_loss = keras.losses.mean_absolute_error(y_true, y_pred)

    return K.mean(-K.sum(_y_true * log_softmax, axis=-1)) + cfg.λ * l1_loss


def sparse_pixelwise_accuracy(y_true, y_pred):
    num_class = K.int_shape(y_pred)[-1]
    _y_pred = K.reshape(y_pred, (-1, num_class))
    _y_true = K.flatten(y_true)

    return K.mean(K.equal(_y_true, tf.to_float(K.argmax(_y_pred, axis=-1))))


def fcn32s(input_shape=(None, None, 3), num_class=5, weights=None):

    img_input = Input(shape=input_shape)
    tf.summary.image('input', img_input, max_outputs=8)

    x = ZeroPadding2D(padding=(100, 100), name='pad1')(img_input)
    x, _ = vgg16(x)
    x = classifier(x, num_class)
    x = Conv2DTranspose(
        filters=num_class,
        kernel_size=(64, 64), strides=(32, 32),
        kernel_initializer='he_normal',
        padding='valid', use_bias=False, name='upscore_lsun')(x)
    x = Cropping2D(cropping=((22, 22), (22, 22)), name='score_lsun')(x)

    model = keras.models.Model(img_input, x, name='fcn32s')

    weights_path = (
        weights or
        get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG_WEIGHTS_NO_TOP_PATH, cache_subdir='models'))  # noqa
    print('==> Using weights:', weights_path)
    model.load_weights(weights_path, by_name=True)

    return model


def fcn16s(input_shape=(None, None, 3), num_class=5, weights=None):

    img_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(100, 100), name='pad1')(img_input)
    x, pool4 = vgg16(x)
    x = classifier(x, num_class)

    x = Conv2DTranspose(
        filters=num_class,
        kernel_size=(4, 4), strides=(2, 2),
        kernel_initializer='he_normal',
        padding='valid', name='upscore_lsun')(x)

    pool4 = Conv2D(
        filters=num_class,
        kernel_size=(1, 1),
        kernel_initializer='he_normal',
        padding='valid', name='pool4_score')(pool4)
    pool4_score = _crop(x, offset=(5, 5), name='pool4_score2')(pool4)
    m = merge([pool4_score, x], mode='sum')

    upscore = Conv2DTranspose(
        filters=num_class,
        kernel_size=(32, 32), strides=(16, 16),
        padding='valid', name='merged_score')(m)

    score = _crop(img_input, offset=(27, 27), name='score_lsun')(upscore)

    model = keras.models.Model(img_input, score, name='fcn16s')

    weights_path = (
        weights or
        get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG_WEIGHTS_NO_TOP_PATH, cache_subdir='models'))  # noqa
    print('==> Using weights:', weights_path)
    model.load_weights(weights_path, by_name=True)

    return model


def classifier(x, num_class):

    conv = partial(Conv2D, activation='relu', padding='valid',
                   kernel_regularizer=keras.regularizers.l2(0.00005))

    x = conv(filters=4096, kernel_size=(7, 7), name='fc6')(x)
    x = Dropout(0.85)(x)
    x = conv(filters=4096, kernel_size=(1, 1), name='fc7')(x)
    x = Dropout(0.85)(x)

    x = Conv2D(
        filters=num_class,
        kernel_size=(1, 1), strides=(1, 1),
        kernel_initializer='he_normal',
        padding='valid', name='score_fr_lsun')(x)
    return x


def vgg16(x):

    conv = partial(Conv2D, activation='relu', padding='same')
    pool = partial(MaxPooling2D, padding='same')

    # Block 1
    x = conv(64, (3, 3), name='block1_conv1')(x)
    x = conv(64, (3, 3), name='block1_conv2')(x)
    x = pool(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv(128, (3, 3), name='block2_conv1')(x)
    x = conv(128, (3, 3), name='block2_conv2')(x)
    x = pool((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv(256, (3, 3), name='block3_conv1')(x)
    x = conv(256, (3, 3), name='block3_conv2')(x)
    x = conv(256, (3, 3), name='block3_conv3')(x)
    x = pool((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv(512, (3, 3), name='block4_conv1')(x)
    x = conv(512, (3, 3), name='block4_conv2')(x)
    x = conv(512, (3, 3), name='block4_conv3')(x)
    x = pool((2, 2), strides=(2, 2), name='block4_pool')(x)
    pool4 = x

    # Block 5
    x = conv(512, (3, 3), name='block5_conv1')(x)
    x = conv(512, (3, 3), name='block5_conv2')(x)
    x = conv(512, (3, 3), name='block5_conv3')(x)
    x = pool((2, 2), strides=(2, 2), name='block5_pool')(x)

    return x, pool4


def _crop(target_layer, offset=(None, None), name=None):
    """Crop the bottom such that it has the same shape as target_layer."""
    def f(input):
        width = input._keras_shape[1]
        height = input._keras_shape[2]
        target_width = target_layer._keras_shape[1]
        target_height = target_layer._keras_shape[2]
        cropped = Cropping2D(
            cropping=(
                (offset[0], width - offset[0] - target_width),
                (offset[1], height - offset[1] - target_height)),
            name='{}'.format(name))(input)
        return cropped
    return f
