from multiprocessing import Pool
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import argparse
from easydict import EasyDict
import yaml
from copy import deepcopy
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score

from utils.feature_util import get_feature_sample_from_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    
    parser.add_argument('--cfg', help='train configure file name', required=False, type=str,default='./cfgs/train_cfg_rf.yaml')
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.cfg) as f:
    cfgs = EasyDict(yaml.load(f))
    

def get_features_all(p):
    i,row = p
    fn = row['filename']
    curr_cls = row['gt'].upper()

    pkl_path = cfgs.pkl_root + '/' + fn + '_dir/' + fn + '.pkl'
    
    with open(pkl_path,'rb') as f:
        pkl_result = pickle.load(f)
    tmp_features,fea_names = get_feature_sample_from_dict(cfgs, pkl_result, result_key="result")    
    if tmp_features is not None:
        tmp_wsi_cls = np.zeros((len(cfgs.wsi_cls_list_train_val),))
        tmp_wsi_cls[cfgs.wsi_cls_list_train_val.index(curr_cls)] = 1
        return [tmp_features,tmp_wsi_cls,os.path.basename(pkl_path)[:-4],fea_names]
    else:
        return []


def get_feature(root,df):
    features = []
    wsicls = []
    wsiname = []  

    p = Pool(24)
    for params in tqdm(p.imap(get_features_all,df.iterrows())):
        if len(params)>0:
            tmp_features,tmp_wsi_cls,filename,fea_names = params
            features.append(tmp_features)
            wsicls.append(tmp_wsi_cls)
            wsiname.append(filename)                

    return np.array(features),np.array(wsicls), np.array(wsiname),fea_names


def main():
    train_df = pd.read_csv(cfgs.rf_train)
    val_df = pd.read_csv(cfgs.rf_val)

    train_features,train_cls,train_wsiname,_ = get_feature(cfgs.pkl_root,train_df)
    val_features,val_cls,val_wsiname,_ = get_feature(cfgs.pkl_root,val_df)
    print(val_features.shape)
    clf = RandomForestClassifier(
        n_estimators=1000, max_depth=18, random_state=0,min_samples_leaf=2,warm_start=False,criterion='gini',#4
        class_weight={0:1.0, 1:2.3, 2:1.5, 3:1.5, 4:2}
        )
    clf.fit(train_features,np.argmax(train_cls,axis=1))
    pickle.dump(clf,open(cfgs.model_path,'wb'))
    proba = clf.predict_proba(train_features)

    print('========= train data report =========')
    cm = confusion_matrix(np.argmax(train_cls,axis=1),np.argmax(proba,axis=1),labels=[0,1,2,3,4])
    print(cm)
    print(classification_report(np.argmax(train_cls,axis=1),np.argmax(proba,axis=1),digits=4) )

    print('\n========= val data report after thresholding =========')
    proba = clf.predict_proba(val_features)
    pred = np.argmax(proba,axis=1) 
    cm = confusion_matrix(np.argmax(val_cls,axis=1),pred,labels=[0,1,2,3,4])
    print(cm)
    print(classification_report(np.argmax(val_cls,axis=1),pred,digits=4,
                                    ) )


def test():
    test_df = pd.read_csv(cfgs.rf_val)
    test_features,test_cls,test_wsiname,fea_names = get_feature(cfgs.pkl_root,test_df)
    
    clf = pickle.load(open(cfgs.model_path,'rb'))
    
    print('\n========= test data report after thresholding =========')
    proba = clf.predict_proba(test_features)
    pred = np.argmax(proba,axis=1) 
    cm = confusion_matrix(np.argmax(test_cls,axis=1),pred,labels=[0,1,2,3,4])
    print(cm)
    print(classification_report(np.argmax(test_cls,axis=1),pred,digits=4,
                                    ) )

if __name__ == '__main__':
    if cfgs.do_train:
        main()
    else:
        test()