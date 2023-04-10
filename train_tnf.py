import argparse
import numpy as np
import yaml
import os
from datetime import datetime
from easydict import EasyDict
import pandas as pd
import pickle
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.keras.engine import data_adapter

from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg', help='train configure file name', required=False, type=str,default='./cfgs/train_cfg_rf.yaml')
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.cfg) as f:
    cfgs = EasyDict(yaml.load(f))


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

BAG_SIZE = cfgs.use_cls_bag_size*len(cfgs.cls_list)
class_mapping = {'yin':0, 'aus':1, 'fn':2, 'ptc':3, 'ot':4, 'mtc':4, 'bfn':0,
        'none':0,'red':0,'nilm':0,'sptc':3, 'hurthle':2, 'scc':4,'ca':4,'utc':4,'adc':4} 
class_weight = {0:1.0,1:1.0,2:1.0,3:1.0,4:1.0}


class MILAttentionLayer(layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def get_config(self):
    
        config = super().get_config().copy()
        config.update({
            'weight_params_dim': self.weight_params_dim,
            'kernel_regularizer':self.kernel_regularizer,
            'use_gated':self.use_gated,
        })
        return config

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[0][1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:

            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(instance, self.w_weight_params, axes=1)
    



def create_model(fea_size):
    
    # Extract features from inputs.
    inputs, embeddings = [], []

    for i_i in range(BAG_SIZE):
        inp = layers.Input((fea_size,))
        inputs.append(inp)

    shared_dense_layer_3 = layers.Dense(128, activation="relu")
    bn_layer_2 = layers.BatchNormalization()
    shared_dense_layer_4 = layers.Dense(128, activation="relu")

    for i_cls in range(len(cfgs.cls_list)):
        shared_dense_layer_1 = layers.Dense(256, activation="relu")
        bn_layer = layers.BatchNormalization()
        shared_dense_layer_2 = layers.Dense(256, activation="relu")
        embeddings_cls = []
        for n_i in range(cfgs.use_cls_bag_size):
            dense_1 = shared_dense_layer_1(layers.Dropout(0.1)(inputs[i_cls*cfgs.use_cls_bag_size+n_i]))
            bn_1 = bn_layer(dense_1)
            bn_1 = layers.Dropout(0.1)(bn_1)
            dense_2 = shared_dense_layer_2(bn_1)
            dense_2 = layers.Dropout(0.1)(dense_2)
            embeddings_cls.append(dense_2)              

        alpha_cls = MILAttentionLayer(
            weight_params_dim=64,
            kernel_regularizer=keras.regularizers.l2(0.001),
            use_gated=True,
        )(embeddings_cls)
        

        # Multiply attention weights with the input layers.
        multiply_layers_cls = [
            layers.multiply([alpha_cls[i], embeddings_cls[i]]) for i in range(len(alpha_cls))
        ]
        
        concat_cls = layers.Add()(multiply_layers_cls)

        dense_3 = shared_dense_layer_3(concat_cls)
        bn_2 = bn_layer_2(dense_3)
        bn_2 = layers.Dropout(0.1)(bn_2)
        dense_4 = shared_dense_layer_4(bn_2)
        dense_4 = layers.Dropout(0.1)(dense_4)

        embeddings.append(dense_4)


    # Invoke the attention layer.
    alpha = MILAttentionLayer(
        weight_params_dim=64,
        kernel_regularizer=keras.regularizers.l2(0.001),
        use_gated=True,
        name="alpha",
    )(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # Concatenate layers.
    concat = layers.concatenate(multiply_layers, axis=1)

    # Classification output node.
    output = layers.Dense(5, activation="softmax")(concat)

    return keras.Model(inputs, output)

def datagen(df_all,pkl_root,is_train):
    if type(pkl_root)!=str:
        pkl_root=pkl_root.decode('UTF-8')
    df_all = pd.DataFrame(df_all)
    df_all = shuffle(df_all)
    
    df = df_all

    for i,row in df.iterrows():
        filename = row[0].decode('UTF-8')
        pkl_path = os.path.join(pkl_root,filename+'_dir',filename+'.pkl')
        if not os.path.exists(pkl_path):
            continue
        y = np.zeros((len(cfgs.slide_class),))
        class_id = class_mapping[row[1].decode('UTF-8').lower()]
        y[class_id] = 1
        y = np.array(y,dtype=np.float32)
        w = 1.0
        yield {'path':pkl_path,'y':y,'w':pd.Series(w),'is_train':is_train}

def get_feature_map_from_pkl(pkl_path,is_train):
    if not isinstance(pkl_path,str):
        pkl_path = pkl_path.numpy()
    if type(pkl_path)!=str:
        pkl_path=pkl_path.decode('UTF-8')
    if not os.path.exists(pkl_path):
        print(pkl_path,'not exists')
    try:
        with open(pkl_path,'rb') as f:
            fea = pickle.load(f)['fea_list_2']
    except Exception as e:
        print(pkl_path,'=============================')
    fea_all = []
    for cls_i in range(len(cfgs.cls_list)):
        if is_train:
            tmp_fea_range = list(range(cls_i*cfgs.cls_bag_size,(cls_i+1)*cfgs.cls_bag_size))[:cfgs.top_cls_bag_size]
            tmp_fea_indx = np.random.choice(tmp_fea_range,cfgs.use_cls_bag_size,replace=False)
            fea_all.extend(fea[tmp_fea_indx])
        else:
            # print('fea',fea.shape,'cls_i', cls_i*cls_bag_size, (cls_i+1)*cls_bag_size, 'use_cls_bag_size',use_cls_bag_size)
            fea_all.extend(fea[cls_i*cfgs.cls_bag_size:(cls_i+1)*cfgs.cls_bag_size][:cfgs.use_cls_bag_size])
    return tuple(fea_all)

def preprocess_image(params):
    npz_path = params['path']
    y = params['y']    
    w = params['w']
    is_train = params['is_train']
    image = tf.py_function(func=get_feature_map_from_pkl,inp=[npz_path,is_train],Tout=tuple([tf.float32]*BAG_SIZE))
    
    y.set_shape([len(cfgs.slide_class),])
    # print
    if cfgs.do_train:
        return tuple(image),y,w
    else:
        return tuple(image),y,npz_path

def train():    
    model = create_model(1280)

    optimizer = tfa.optimizers.RectifiedAdam(lr=0.0001)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(
            optimizer=optimizer, loss=loss, metrics=['accuracy'],
        )

    train_df = pd.read_csv(cfgs.tnf_train)
    val_df = pd.read_csv(cfgs.tnf_val)


    train_ds = tf.data.Dataset.from_generator(datagen, output_types={'path':tf.string,'y':tf.float32, 'w':tf.float32,'is_train':tf.bool},
                                args=(train_df,cfgs.pkl_root,True)).map(preprocess_image,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE).batch(cfgs.batch_size)
    val_ds = tf.data.Dataset.from_generator(datagen, output_types={'path':tf.string,'y':tf.float32, 'w':tf.float32,'is_train':tf.bool},
                                args=(val_df,cfgs.pkl_root,False)).map(preprocess_image,
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE).batch(len(cfgs.test_df))
    

    filepath = cfgs.ckpt_dir + "/weights-{epoch:03d}-{val_accuracy:.4f}.hdf5"
    log_dir = cfgs.ckpt_dir + "/log/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint = ModelCheckpoint(filepath, verbose=1,save_best_only=False,mode='max')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,update_freq='batch')

    model.fit(train_ds,epochs=1000,validation_data=val_ds,callbacks=[checkpoint,tensorboard_callback],workers=4, 
                    use_multiprocessing=True,validation_freq=1)
    
class TestModel(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x = data_adapter.unpack_x_y_sample_weight(data)
        im = x[0]
        y_true = x[1]
        filename = x[2]
        output =  self(im, training=False)
        return (output,y_true,filename)


def run_test(ckpt_path):
    test_df = pd.read_csv(cfgs.tnf_test)

    test_ds = tf.data.Dataset.from_generator(datagen, output_types={'path':tf.string,'y':tf.float32, 'w':tf.float32,'is_train':tf.bool},
                             args=(test_df,cfgs.pkl_root,False)).map(preprocess_image,
                                                                     num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(128).batch(cfgs.batch_size)

    model = tf.keras.models.load_model(ckpt_path,custom_objects={'relu6':tf.nn.relu6,
                                'MILAttentionLayer':MILAttentionLayer,
                                'L2':keras.regularizers.l2,
                                },compile=False)
    model = TestModel(inputs=model.input,outputs=model.output)
    # model.summary()

    y_true =[]
    proba = []

    res = model.predict(test_ds,verbose=1)
    proba,y_true,wsiname = res
    y_true = np.argmax(y_true,axis=1)
    pred = np.argmax(proba,axis=1) 

    wsiname = [os.path.basename(i.decode('UTF-8')) for i in wsiname]


    cm = confusion_matrix(y_true,pred,labels=[0,1,2,3,4])
    print(cm)
    print(classification_report(y_true,pred,digits=4,) )

