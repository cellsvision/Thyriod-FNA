from glob import glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def get_wsi_path(root_path,class_list=None):

    if class_list is None:
        WSIdir = list(glob(root_path + '/*'))
        WSIclass = ['unk']*len(WSIdir)
        
    else:
        WSIclass = []
        WSIdir = []
        for i,c in enumerate(class_list):
            wsis = list(glob(root_path + '/' + c + '/*'))
            WSIdir.extend(wsis)
            WSIclass.extend([c]*len(wsis))

    df = pd.DataFrame({
        'wsi_dir': WSIdir,
        'wsi_cls': WSIclass
    })
    return df


def get_file_path(df,cfgs):
    wsi_dir =  np.array(df['wsi_dir'])
    image_pathes = []
    patch_label = []  # ptc,susptc...
    patch_cls = [] # ptc,bfn...
    for wsi in wsi_dir:
        for curr_cls,curr_labels in cfgs.train_patch_class_mapping.items():
            for curr_label in curr_labels:
                files = list(glob(wsi + '/*/*_' + curr_label + '.png'))
                if len(files)>0:  
                    image_pathes.extend(files)
                    patch_cls.extend([curr_cls]*len(files))
                    patch_label.extend([curr_label]*len(files))

        if len(list(glob(wsi + '/others/*.png')))>0:
            image_pathes.append(wsi + '/others')
            patch_cls.append('others')
            patch_label.append('bg_others')

    new_df = pd.DataFrame()
    new_df['image_pathes']  = image_pathes
    new_df['patch_label'] = patch_label
    new_df['patch_cls'] = patch_cls
    new_df = shuffle(new_df)
    return new_df





