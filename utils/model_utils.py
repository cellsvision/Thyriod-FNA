import tensorflow as tf 
import tensorflow.keras as keras 
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.pooling import GlobalPooling2D


preprocessing_func_dict = {
    'efficientnet':keras.applications.efficientnet.preprocess_input,
    'resnet_v2': keras.applications.resnet_v2.preprocess_input,
    'resnet50': keras.applications.resnet50.preprocess_input,
    'densenet': keras.applications.densenet.preprocess_input,
    'vgg16': keras.applications.vgg16.preprocess_input,
    'inception-v3':keras.applications.inception_v3.preprocess_input,
}

basemodel_dict = {
    'EfficientNetB0':  keras.applications.EfficientNetB0,
    'EfficientNetB1':  keras.applications.EfficientNetB1,
    'EfficientNetB2':  keras.applications.EfficientNetB2,
    'ResNet50': keras.applications.ResNet50,
    'ResNet50V2': keras.applications.ResNet50V2,
    'VGG16': keras.applications.VGG16,
    'Inception-3':keras.applications.InceptionV3,
}

scheduler_dict = {
    'CosineDecayRestarts': tf.keras.experimental.CosineDecayRestarts,
    'NoisyLinearCosineDecay': tf.keras.experimental.NoisyLinearCosineDecay,
    'CosineDecay': tf.keras.experimental.CosineDecay,
    'LinearCosineDecay': tf.keras.experimental.LinearCosineDecay
}

class LocalImportancebasedPooling(GlobalPooling2D):
    def call(self, inputs,logits=None):
        if logits is None:
            weights = tf.exp(inputs)
        else:
            weights = tf.exp(logits)

        if self.data_format == 'channels_last':
            return K.mean(inputs*weights, axis=[1, 2]) / K.mean(weights, axis=[1, 2])
        else:
            return K.mean(inputs*weights, axis=[2, 3]) / K.mean(weights, axis=[2, 3])


def channel_attention(input_feature, ratio=8):        
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = keras.layers.Dense(channel//ratio,
                                activation='relu',
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
    shared_layer_two = keras.layers.Dense(channel,
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')

    avg_pool = keras.layers.GlobalAveragePooling2D()(input_feature)    
    avg_pool = keras.layers.Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)

    max_pool = keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = keras.layers.Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)

    cbam_feature = keras.layers.Add()([avg_pool,max_pool])
    cbam_feature = keras.layers.Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = tf.keras.layers.Permute((3, 1, 2))(cbam_feature)

    return keras.layers.Multiply()([input_feature, cbam_feature]) 

def spatial_attention(input_feature,confmap=None):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = tf.keras.layers.Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = tf.keras.layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = tf.keras.layers.Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = tf.keras.layers.Permute((3, 1, 2))(cbam_feature)
    if confmap is not None:
        tf.keras.layers.Multiply()([confmap, cbam_feature])
    else:    
        return tf.keras.layers.Multiply()([input_feature, cbam_feature])
    

def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def get_model(cfgs):
    basemodel = basemodel_dict[cfgs['basemodel']]

    base_model = basemodel(
        include_top=False, weights='imagenet',
        input_shape=(cfgs.model_input_size,cfgs.model_input_size,3),
        pooling=None,)
    model_input = base_model.input
    baseout = base_model.output

    x = keras.layers.BatchNormalization()(baseout)

    x = cbam_block(x,ratio=4)
    
    x = keras.layers.Conv2D(filters=len(cfgs.train_patch_class_list),kernel_size=1,strides=1, use_bias=False,padding='same',name='att_layer')(x)
    logits = keras.layers.Conv2D(filters=len(cfgs.train_patch_class_list),kernel_size=3,strides=1, use_bias=False,padding='same',name='att_layer_2')(x)
    x = LocalImportancebasedPooling()(x,logits)
    x = keras.layers.Softmax()(x)

    model = keras.models.Model(inputs=model_input,outputs=x)
    return model
