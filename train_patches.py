import os
import argparse
import yaml

from easydict import EasyDict
import pandas as pd
from datetime import datetime

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



if __name__ == "__main__":       
    train_fns_df = pd.read_csv(cfgs.train_csv_path,dtype=str)['filename'].values
    val_fns_df = pd.read_csv(cfgs.val_csv_path,dtype=str)['filename'].values

    with strategy.scope():
        root_path = cfgs.patches_root
        class_list = cfgs.train_class_list

        WSI_df = data_utils.get_wsi_path(root_path,class_list)
        train_df = WSI_df[WSI_df['wsi_dir'].apply(lambda x: os.path.basename(x)).isin(train_fns_df)]
        test_df = WSI_df[WSI_df['wsi_dir'].apply(lambda x: os.path.basename(x)).isin(val_fns_df)]


        train_df = data_utils.get_file_path(train_df,cfgs)
        test_df = data_utils.get_file_path(test_df,cfgs)

        print('data preparation done.')

        foldername,labels = train_df['image_pathes'].values,train_df['patch_cls'].values
        foldername_te,labels_te = test_df['image_pathes'].values,test_df['patch_cls'].values

        train_ds = tf.data.Dataset.from_generator(data_generators.tfd_patch(train_df.columns,'train',cfgs), output_types={'im': tf.string,'labels':tf.float32,'weight':tf.float32},
                             args=(train_df,)).map(data_generators.mf_tfd_patch(cfgs),
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(cfgs.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = tf.data.Dataset.from_generator(data_generators.tfd_patch(test_df.columns,'val',cfgs), output_types={'im': tf.string,'labels':tf.float32,'weight':tf.float32},
                            args=(test_df,)).map(data_generators.mf_tfd_patch(cfgs),
                                                num_parallel_calls=8).batch(3).prefetch(tf.data.experimental.AUTOTUNE)

        model = model_utils.get_model(cfgs)
        # model.summary()


        if cfgs.optimizer == 'adam':
            optim = keras.optimizers.Adam
        elif cfgs.optimizer == 'radam':
            optim = tfa.optimizers.RectifiedAdam
        elif cfgs.optimizer == 'sgd':
            optim = keras.optimizers.SGD
        else:
            raise NotImplementedError
        
        if cfgs.scheduler is not None:
            cfgs.scheduler_args['initial_learning_rate'] = cfgs.LR
            scheduler = model_utils.scheduler_dict[cfgs.scheduler](**cfgs.scheduler_args) 
            optimizer =  optim(learning_rate=scheduler,**cfgs.optim_args)
        else:
            optimizer =  optim(lr=cfgs.LR)

        loss = categorical_focal_loss()   
        model.compile(optimizer=optimizer,metrics=['accuracy'],loss=loss)

        filepath = cfgs.save_model_path + "/weights-{epoch:03d}-{accuracy:.4f}_{val_accuracy:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=False)

        log_dir = cfgs.save_model_path + "/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1,update_freq='batch')
        callbacks_list = [tensorboard_callback,checkpoint] 

        model.fit(train_ds,epochs=500,validation_data=val_ds,callbacks=callbacks_list)






    print('finished')