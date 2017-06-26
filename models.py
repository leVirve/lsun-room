from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dropout, Permute, Add, add
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.layers.core import Reshape, Flatten, Activation
from keras.utils.data_utils import get_file
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.regularizers import l1,l2
from keras.layers.core import ActivityRegularization
from keras.initializers import RandomUniform
from resnet_blocks import *
from cropping import *

VGG_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
RES_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def _crop(target_layer, offset=(None, None), name=None):
    """Crop the bottom such that it has the same shape as target_layer."""
    def f(input):
        width = input._keras_shape[1]
        height = input._keras_shape[2]
        target_width = target_layer._keras_shape[1]
        target_height = target_layer._keras_shape[2]
        cropped = Cropping2D(cropping=((offset[0], width - offset[0] - target_width),(offset[1],height - offset[1] - target_height)), name='{}'.format(name))(input)
        return cropped
    return f

def fcn_vggbase(input_shape=(None,None,3)):
    
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(100, 100), name='pad1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='block5_pool')(x)

    x = Conv2D(filters=4096, kernel_size=(7, 7), W_regularizer=l2(0.00005), activation='relu', padding='valid', name='fc6_lsun')(x)
    x = Dropout(0.85)(x)
    x = Conv2D(filters=4096, kernel_size=(1, 1), W_regularizer=l2(0.00005), activation='relu', padding='valid', name='fc7_lsun')(x)
    x = Dropout(0.85)(x)
    x = Conv2D(filters=5, kernel_size=(1, 1), strides=(1,1), kernel_initializer='he_normal', padding='valid', name='lsun_score')(x)

    x = Conv2DTranspose(filters=5, kernel_initializer='he_normal', kernel_size=(64, 64), strides=(32, 32), padding='valid',use_bias=False, name='lsun_upscore2')(x)
    output = _crop(img_input,offset=(32,32), name='score')(x)

    model = Model(img_input, output)
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)

    return model

def fcn16s_vggbase(input_shape=None, nb_class=None):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(100, 100), name='pad1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
          
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)
         
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)
         
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)
    pool4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='block5_pool')(x)
    
    x = Conv2D(filters=4096, kernel_size=(7, 7), W_regularizer=l2(0.00005), activation='relu', padding='valid', name='fc6')(x)
    x = Dropout(0.85)(x)
    x = Conv2D(filters=4096, kernel_size=(1, 1), W_regularizer=l2(0.00005), activation='relu', padding='valid', name='fc7')(x)
    x = Dropout(0.85)(x)
    x = Conv2D(filters=nb_class, kernel_size=(1, 1), strides=(1,1), kernel_initializer='he_normal', padding='valid', name='p5score')(x)
    x = Conv2DTranspose(filters=nb_class, kernel_size=(4,4), strides=(2,2), kernel_initializer='he_normal', padding='valid', name='p5upscore')(x)

    pool4 = Conv2D(filters=nb_class, kernel_size=(1,1), kernel_initializer='he_normal', padding='valid', name='pool4_score')(pool4)
    pool4_score = _crop(x, offset=(5,5), name='pool4_score2')(pool4)
    m = merge([pool4_score,x], mode='sum')
    upscore = Conv2DTranspose(filters=nb_class, kernel_size=(32,32), strides=(16,16), padding='valid', name='merged_score')(m)
    score = _crop(img_input, offset=(27,27), name='output_score')(upscore)

    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    mdl = Model(img_input, score, name='fcn16s')
    mdl.load_weights(weights_path, by_name=True)

    return mdl

def fcn_Resnet50(input_shape = None, weight_decay=0.0002, batch_momentum=0.9, batch_shape=None, classes=40):

    img_input = Input(shape=input_shape)
    bn_axis = 3

    x = Conv2D(64, kernel_size=(7,7), subsample=(2, 2), border_mode='same', name='conv1', W_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(3, [64, 64, 256], stage=2, block='a', strides=(1, 1))(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='b')(x)
    x = identity_block(3, [64, 64, 256], stage=2, block='c')(x)

    x = conv_block(3, [128, 128, 512], stage=3, block='a')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='b')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='c')(x)
    x = identity_block(3, [128, 128, 512], stage=3, block='d')(x)

    x = conv_block(3, [256, 256, 1024], stage=4, block='a')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='b')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='c')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='d')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='e')(x)
    x = identity_block(3, [256, 256, 1024], stage=4, block='f')(x)

    x = conv_block(3, [512, 512, 2048], stage=5, block='a')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='b')(x)
    x = identity_block(3, [512, 512, 2048], stage=5, block='c')(x)
    #classifying layer
    x = Conv2D(filters=40, kernel_size=(1,1), strides=(1,1), init='he_normal', activation='linear', border_mode='valid', W_regularizer=l2(weight_decay))(x)

    x = Conv2DTranspose(filters=40, kernel_initializer='he_normal', kernel_size=(64, 64), strides=(32, 32), padding='valid',use_bias=False, name='upscore2')(x)
    x = Cropping2D(cropping=((19, 36),(19, 29)), name='score')(x)

    model = Model(img_input, x)
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', RES_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)
    
    return model

def dilated_FCN_addmodule(input_shape=None):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(100, 100), name='pad1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same',name='block5_pool')(x)

    x = Conv2D(filters=4096, kernel_initializer='he_normal', kernel_size=(7, 7), activation='relu', padding='valid', name='fc6')(x)
    x = Dropout(0.85)(x)
    x = Conv2D(filters=4096, kernel_initializer='he_normal', kernel_size=(1, 1), activation='relu', padding='valid', name='fc7')(x)
    x = Dropout(0.85)(x)
    x = Conv2D(filters=40,kernel_size=(1, 1), strides=(1,1), kernel_initializer='he_normal', padding='valid', name='score_fr')(x)
    #x = Cropping2D(cropping=((19, 36),(19, 29)), name='score')(x)
    x = ZeroPadding2D(padding=(33,33))(x)
    x = Conv2D(2*40, (3,3), kernel_initializer='he_normal',activation='relu', name='dl_conv1')(x)
    x = Conv2D(2*40, (3,3), kernel_initializer='he_normal',activation='relu', name='dl_conv2')(x)
    x = Conv2D(4*40, (3,3), kernel_initializer='he_normal',dilation_rate=(2,2), activation='relu', name='dl_conv3')(x)
    x = Conv2D(8*40, (3,3), kernel_initializer='he_normal',dilation_rate=(4,4), activation='relu', name='dl_conv4')(x)
    x = Conv2D(16*40, (3,3), kernel_initializer='he_normal',dilation_rate=(8,8), activation='relu', name='dl_conv5')(x)
    x = Conv2D(32*40, (3,3), kernel_initializer='he_normal',dilation_rate=(16,16), activation='relu', name='dl_conv6')(x)
    x = Conv2D(32*40, (1,1), kernel_initializer='he_normal',name='dl_conv7')(x)
    x = Conv2D(1*40, (1,1), kernel_initializer='he_normal',name='dl_final')(x)
    x = Conv2DTranspose(filters=40, kernel_initializer='he_normal', kernel_size=(64, 64), strides=(32, 32), padding='valid',use_bias=False, name='upscore2')(x)
    x = CroppingLike2D(img_input, offset='centered', name='score')(x)

    mdl = Model(img_input, x, name='dilatedmoduleFCN')
    #weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    mdl.load_weights('logs/model_June13_sgd_60kitr.h5', by_name=True)
    return mdl

def dilated_FCN_frontended(input_shape=None, weight_decay=None, nb_classes=40):

    img_input = Input(shape=input_shape)

    #x = ZeroPadding2D(padding=(100, 100), name='pad1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    
    x = Conv2D(512, (3,3), dilation_rate=(2,2), activation='relu', name='block5_conv1')(x)
    x = Conv2D(512, (3,3), dilation_rate=(2,2), activation='relu', name='block5_conv2')(x)
    x = Conv2D(512, (3,3), dilation_rate=(2,2), activation='relu', name='block5_conv3')(x)
    
    x = Conv2D(4096, (3,3), kernel_initializer='he_normal', dilation_rate=(4,4), activation='relu', name='fc6')(x)
    x = Dropout(0.5, name='drop6')(x)
    x = Conv2D(4096, (1,1), kernel_initializer='he_normal', activation='relu', name='fc7')(x)
    x = Dropout(0.5, name='drop7')(x)
    x = Conv2D(nb_classes, (1,1), kernel_initializer='he_normal', activation='relu', name='fc_final')(x)
    

    #x = Conv2DTranspose(nb_classes, kernel_size=(64,64), strides=(32,32), padding='valid', name='upscore2')(x)    
    x = ZeroPadding2D(padding=(33,33))(x)
    x = Conv2D(2*nb_classes, (3,3), kernel_initializer='he_normal',activation='relu', name='dl_conv1')(x)
    x = Conv2D(2*nb_classes, (3,3), kernel_initializer='he_normal',activation='relu', name='dl_conv2')(x)
    x = Conv2D(4*nb_classes, (3,3), kernel_initializer='he_normal',dilation_rate=(2,2), activation='relu', name='dl_conv3')(x)
    x = Conv2D(8*nb_classes, (3,3), kernel_initializer='he_normal',dilation_rate=(4,4), activation='relu', name='dl_conv4')(x)
    x = Conv2D(16*nb_classes, (3,3), kernel_initializer='he_normal',dilation_rate=(8,8), activation='relu', name='dl_conv5')(x)
    x = Conv2D(32*nb_classes, (3,3), kernel_initializer='he_normal',dilation_rate=(16,16), activation='relu', name='dl_conv6')(x)
    x = Conv2D(32*nb_classes, (1,1), kernel_initializer='he_normal',name='dl_conv7')(x)
    x = Conv2D(1*nb_classes, (1,1), kernel_initializer='he_normal',name='dl_final')(x)
    x = Conv2DTranspose(nb_classes, kernel_initializer='he_normal', kernel_size=(64,64), strides=(8,8), padding='valid', name='upscore2')(x)
    x = CroppingLike2D(img_input, offset='centered', name='score')(x)
    #x = Cropping2D(cropping=((19,36), (19,29)), name='score')(x)
   

    mdl = Model(input=img_input, output=x, name='dilated_fcn')
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    mdl.load_weights(weights_path, by_name=True)
    return mdl

def dilat_fets(input_shape=None, classes=40):
    
    model_in = Input(shape=input_shape)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(model_in)
    h = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(h)
    
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(h)
    h = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(h)
    
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(h)
    h = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(h)
    h = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(h)
    
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(h)
    h = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(h)
    
    h = AtrousConvolution2D(512, 3, 3, dilation_rate=(2, 2), activation='relu', name='conv5_1')(h)
    h = AtrousConvolution2D(512, 3, 3, dilation_rate=(2, 2), activation='relu', name='conv5_2')(h)
    h = AtrousConvolution2D(512, 3, 3, dilation_rate=(2, 2), activation='relu', name='conv5_3')(h)
    
    h = AtrousConvolution2D(4096, 7, 7, dilation_rate=(4, 4), activation='relu', name='fc6')(h)
    h = Dropout(0.5, name='drop6')(h)
    h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
    h = Dropout(0.5, name='drop7')(h)
    h = Convolution2D(classes, 1, 1, activation='relu', name='fc-final')(h)
    
    h = ZeroPadding2D(padding=(33, 33))(h)
    
    h = Convolution2D(2 * classes, 3, 3, activation='relu', name='ct_conv1_1')(h)
    h = Convolution2D(2 * classes, 3, 3, activation='relu', name='ct_conv1_2')(h)
    h = AtrousConvolution2D(4 * classes, 3, 3, dilation_rate=(2, 2), activation='relu', name='ct_conv2_1')(h)
    h = AtrousConvolution2D(8 * classes, 3, 3, dilation_rate=(4, 4), activation='relu', name='ct_conv3_1')(h)
    h = AtrousConvolution2D(16 * classes, 3, 3, dilation_rate=(8, 8), activation='relu', name='ct_conv4_1')(h)
    h = AtrousConvolution2D(32 * classes, 3, 3, dilation_rate=(16, 16), activation='relu', name='ct_conv5_1')(h)
    h = Convolution2D(32 * classes, 3, 3, activation='relu', name='ct_fc1')(h)
    h = Convolution2D(classes, 1, 1, name='ct_final')(h)


    model = Model(input=model_in, output=logits, name='dilation_voc12')
    return model
