import os
import pandas as pd
import numpy as np
from glob import glob
import random
import albumentations as A
from PIL import Image
from datetime import datetime
from sklearn.utils import shuffle
import cv2

import tensorflow as tf

from .model_utils import preprocessing_func_dict

def get_tmp_df(df,phase,cfgs):
    sample_weight = cfgs.sample_weight

    if phase=='train' or phase=='val':
        tmp_filepath = []
        tmp_cls = []
        tmp_w = []
        for i,row in df.iterrows():
            if row['patch_label'] == 'bg_others':
                tmp_f = glob(row['image_pathes'] + '/*.png')
                tmp_f = np.random.choice(tmp_f,min(cfgs.select_n_others,len(tmp_f)),replace=False)
                tmp_filepath.extend(tmp_f)
                tmp_cls.extend([row['patch_cls']]*len(tmp_f))
                curr_w = sample_weight['bg_others'] if 'bg_others' in sample_weight.keys() else 1.0
                tmp_w.extend([curr_w]*len(tmp_f))
            
            else:
                if phase=='val':
                    tmp_f = row['image_pathes'][:-4] + '.png'
                    tmp_filepath.append(tmp_f)
                else:
                    tmp_f = glob(row['image_pathes'][:-4] + '*.png')
                    tmp_f = np.random.choice(tmp_f,1)
                    tmp_filepath.extend(tmp_f)
                tmp_cls.extend([row['patch_cls']]*1)
                curr_label = row['patch_label']
                curr_w = sample_weight[curr_label] if curr_label in sample_weight.keys() else 1.0
                tmp_w.extend([curr_w]*1)
    else:
        raise NotImplementedError

    tmp_df = pd.DataFrame()
    tmp_df['filepath'] = tmp_filepath
    tmp_df['cls'] = tmp_cls
    tmp_df['w'] = tmp_w

    tmp_df = shuffle(tmp_df)

    return tmp_df



# def tfd_patch(df,cols, phase, cfgs):
def tfd_patch(cols, phase, cfgs):
    def f(df):
        assert phase in ['train','val','test'],str(phase)

        df = pd.DataFrame(df)
        df.columns = cols
        for c in cols:
            df[c] = df[c].apply(lambda x:x.decode('UTF-8'))

        if phase == 'train':
            for curr_label,up_times in cfgs.upsample.items():
                dfs = [df] + [df[df['patch_label']==curr_label]] * up_times
                df = pd.concat(dfs)

        tmp_df = get_tmp_df(df,phase,cfgs) 
        print(tmp_df)  
        label_list = cfgs.train_patch_class_list
        for i,rows in tmp_df.iterrows():
            label = np.zeros((len(label_list),),dtype=np.uint8)+ (0.15/(len(label_list)-1))
            assert rows['cls'] in label_list
            label[label_list.index(rows['cls'])] = 0.85
            a = {'im':rows['filepath'],'labels':label,'weight':rows['w']} 
            if rows['cls']!='others':
                for _ in range(cfgs.instances_per_image):
                    yield a
            else:
                yield a
    return f

def create_aug_seq():
    seq = A.Compose([
        A.JpegCompression(p=0.3,quality_lower=70,quality_upper=95),
        A.CoarseDropout(max_holes=8,min_holes=4,max_width=96,max_height=96,min_height=24,min_width=24,p=0.5),
        # A.RandomGridShuffle(always_apply=False, p=0.5, grid=(2, 2)),
        A.OneOf([
            A.RandomGamma(gamma_limit=[50,150],p=0.5),
            A.CLAHE(clip_limit=2.5,tile_grid_size=(80, 80),p=0.5),
            A.RandomBrightnessContrast(p=0.5,brightness_limit=[-0.3,0.3],contrast_limit=[-0.25,0.21]),
        ]),
        A.Flip(p=0.7),
        A.OneOf([
            A.HueSaturationValue(p=0.6,hue_shift_limit=[-10,10],sat_shift_limit=[-45,25],val_shift_limit=[0.0,0.0]),
            # A.ChannelDropout(always_apply=False, p=0.4, channel_drop_range=(1, 1), fill_value=0) 
            A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, always_apply=False, p=0.4),
            # A.ToGray(p=0.3)   
        ]),
        A.OneOf([
            A.ISONoise(p=0.5,color_shift=(0.01, 0.2),intensity=(0.08, 0.13)),
            A.MultiplicativeNoise(p=0.5,multiplier=(0.65, 1.2),per_channel=False,),
            A.GaussNoise(p=0.5,var_limit=(10,150),),
        ])
    ])

    return seq

def read_image(
        filename,
        label,
        apply_aug,
        target_size,
        train_size,
        basemodel_type,
        ):
    train_size = int(train_size)
    if type(filename.numpy())!=str:
        filename=filename.numpy().decode('UTF-8')
    label = label.numpy()
    if not os.path.exists(filename):
        with open('./error_images.txt','w+') as f:
            f.write(datetime.now().strftime("%Y%m%d-%H%M%S") + ':' + filename + '  FileNotFoundError \n\n' )        
        im = np.zeros((target_size,target_size,3),dtype=np.uint8)
        label[0] = 1
        label[1:] = 0
        os.remove(filename)
        return im,label
        
    im = np.array(Image.open(filename))
    if len(im.shape)!=3 or im.shape[-1]!=3:
        with open('./error_images.txt','w+') as f:
            f.write(datetime.now().strftime("%Y%m%d-%H%M%S") + ':' + filename + '   '+ str(im.shape) +'\n\n' )
        im = np.zeros((target_size,target_size,3),dtype=np.uint8)
        label[0] = 1
        label[1:] = 0
        os.remove(filename)
        return im,label
        

    if len(im.shape)!=3 or im.shape[1]<target_size or im.shape[0]<target_size or np.min(im)<0 or np.max(im)>255:
        im = np.zeros((target_size,target_size,3),dtype=np.uint8)
        label[0] = 1
        label[1:] = 0
        with open('./error_images.txt','w+') as f:
            f.write(datetime.now().strftime("%Y%m%d-%H%M%S") + ':' + filename + '\n\n')
        os.remove(filename)
        return im,label
    h,w,_ = im.shape
    
    if (h > target_size) or (w > target_size):
        x = random.randint(0,im.shape[1]-target_size)
        y = random.randint(0,im.shape[0]-target_size)
    
        im = im[y:y+target_size,x:x+target_size]

    try:
        if apply_aug:
            seq = create_aug_seq()
            im = seq(image=im)['image']
    except Exception as e:
        print('======',e,type(seq(image=im)))
        print('========',filename)
        with open('./error_images.txt','w+') as f:
            f.write(datetime.now().strftime("%Y%m%d-%H%M%S") + ':' + str(e) + '\n' + filename + '\n\n')
    assert im.shape == (target_size,target_size,3),str(im.shape)
    im = cv2.resize(im,(train_size,train_size))
    assert im.shape == (train_size,train_size,3),str(im.shape)

    preprocessing_func = preprocessing_func_dict[basemodel_type.numpy().decode('UTF-8')]
    im = preprocessing_func(np.expand_dims(im,0))[0]
    return im,label


def mf_tfd_patch(cfgs):
    def f(params):    
        filename_ = params['im']
        label_ = params['labels']
        weight = params['weight']
        image,label = tf.py_function(func=read_image,
                                     inp=(
                                        filename_, label_,
                                        True,
                                        cfgs.crop_size,
                                        cfgs.model_input_size,
                                        cfgs.basemodel_type
                                        ),
                                     Tout=(tf.float32,tf.float32))
        image.set_shape([cfgs.model_input_size,cfgs.model_input_size,3])
        label.set_shape([len(cfgs.train_patch_class_list)])

        if not cfgs.do_train:
            out = (image,filename_, label)
            return out,[]            
        return image,label
    return f


