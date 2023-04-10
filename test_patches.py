import os
import argparse
import yaml

from easydict import EasyDict
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report,confusion_matrix

from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras.callbacks import Callback, ModelCheckpoint

from utils import model_utils, data_utils, data_generators
from utils.losses import categorical_focal_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg', help='train configure file name', required=False, type=str,default='/home/yaoqy/paper_source_code/cfgs/train_cfg_effi.yaml')
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.cfg) as f:
    cfgs = EasyDict(yaml.load(f))
    cfgs.do_train = True
    
print(f'train cfgs:',cfgs)

if cfgs.no_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.gpu_ids
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy()




class TestModel(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def predict_step(self, data):
        data = data_adapter.expand_1d(data)
        x, _, _ = data_adapter.unpack_x_y_sample_weight(data)
        im = x[0]
        fn = x[1]
        y_true = x[2]
        output =  self(im, training=False)
        return (output,fn,y_true)
    


if __name__ == "__main__":  
    test_fns_df = pd.read_csv(cfgs.test_csv_path,dtype=str)['filename'].values

    with strategy.scope():
        root_path = cfgs.patches_root
        class_list = cfgs.train_class_list

        WSI_df = data_utils.get_wsi_path(root_path,class_list)
        test_df = WSI_df[WSI_df['wsi_dir'].apply(lambda x: os.path.basename(x)).isin(test_fns_df)]

        test_df = data_utils.get_file_path(test_df,cfgs)

        print('data preparation done.')

       
        val_ds = tf.data.Dataset.from_generator(data_generators.tfd_patch(test_df.columns,'val',cfgs), output_types={'im': tf.string,'labels':tf.float32,'weight':tf.float32},
                            args=(test_df,)).map(data_generators.mf_tfd_patch(cfgs),
                                                num_parallel_calls=8).batch(3).prefetch(tf.data.experimental.AUTOTUNE)

        cfgs.upsample = {}


        model = tf.keras.models.load_model(cfgs.load_model_path,custom_objects={'relu6':tf.nn.relu6,
                                            'LocalImportancebasedPooling': model_utils.LocalImportancebasedPooling,
                                            },compile=False)
        model.summary()
        model = TestModel(inputs=model.input,outputs=model.output)

    
        pred = []
        y_true =[]
        fns = []
        proba = []

        
        res = model.predict(val_ds,verbose=1)
        proba,fn,y_true = res
        
        pred = np.argmax(proba,axis=1)
        y_true = np.argmax(y_true,axis=1)
        

        print(confusion_matrix(y_true,pred))
        print(classification_report(y_true,pred,digits=4,target_names=cfgs.train_patch_class_list))





    print('finished')