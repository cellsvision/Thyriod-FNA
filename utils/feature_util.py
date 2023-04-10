import numpy as np
import copy

def filter_each_top_n(result,n,cell_class):
    for i in cell_class:
        if len(result[i])>0:
            tmp_box = result[i]
            result[i] = tmp_box[np.argsort(tmp_box[:,4])[::-1]][:n]
    return result

def filter_all_top_n(result,n,cell_class): 
    all_box = []
    for i in cell_class:
        all_box.extend(result[i])
    all_box = np.array(all_box)
    if len(all_box.shape)<2:
        return result
    all_box = all_box[np.argsort(all_box[:,4])]
    thresh = all_box[:,4][-min(len(all_box),n)]
    for i in cell_class:
        if len(result[i])>0:
            tmp_box = result[i]
            result[i] = tmp_box[tmp_box[:,4]>=thresh]
    return result


def get_hist(
    result,
    cell_class,
    thresh=0.00,
    top_num_all_class=100, # -1
    top_num_each_class=20, # -1
    normalize_axis='vh', # 'v' 'h' 'vh' 'all'
    conf_bins=30,
    conf_range=[0.0,1.0],
    cumsum=False
    ): 
    conf_list = []
    class_list = []

    if top_num_all_class > 0:
        result = filter_all_top_n(result,top_num_all_class,cell_class)
    if top_num_each_class>0:
        result = filter_each_top_n(result,top_num_each_class,cell_class)

    for i,c in enumerate(cell_class):
        c_content = result[c]
        if len(c_content)>0:
            conf_list.extend(c_content[:,4])
            class_list.extend(len(c_content)*[i])

    hist, xedges, yedges = np.histogram2d(conf_list, class_list, bins=[conf_bins,len(cell_class)],range=[conf_range, [0,len(cell_class)]])
    if normalize_axis=='v':
        if cumsum:
            hist = np.cumsum(hist,axis=0)
        hist = hist/(np.sum(hist,axis=0)+10e-5)
    elif normalize_axis=='h':
        if cumsum:
            hist = np.cumsum(hist,axis=0)
        hist = hist.T/(np.sum(hist.T,axis=0)+10e-5).T
        hist = hist.T
    elif normalize_axis=='vh':
        hist = hist/(np.sum(hist)+10e-5)
    else:
        hist = np.cumsum(hist,axis=0)

    return hist

def get_feature_names(prefix,cell_type,shape):
    prefix = prefix + '_'
    c = np.array([list(cell_type)]*shape[0],dtype=np.str)
    p = np.array([[prefix]*shape[1]]*shape[0],dtype=np.str)
    s = np.repeat([np.arange(0,shape[0])],shape[1],axis=0).T

    name = np.core.defchararray.add(p,c)
    name = np.core.defchararray.add(name,s.astype(np.str))
    return name

def filter_size(result, mpp=None):
    for c_cls in ['bfn','aus','ptc','fn','mtc','ss']:
        c_cls_r = result[c_cls]
        rm_index = []
        if len(c_cls_r) > 0:
            for i in range(len(c_cls_r)):
                if ((c_cls_r[i][3] - c_cls_r[i][1]) * mpp / 0.2484 < 15) or ((c_cls_r[i][2] - c_cls_r[i][0]) * mpp / 0.2484 < 15):
                    rm_index.append(i)
            c_cls_r = np.delete(c_cls_r, rm_index, axis=0)
            result[c_cls] = c_cls_r

    return result

def get_feature_sample_from_dict(configs, result_pkl,result_key):
    features_configs = configs['feature_configs']
    cell_cls_list = configs['cell_class']

    result = copy.deepcopy(result_pkl[result_key])
    mpp = float(result_pkl['mpp'])
    tmp_features = []
    features_names = []

    for f in features_configs.keys():
        if features_configs[f]['type'] == 'top_conf':
            for c in cell_cls_list:
                curr_cell = np.array(result[c])
                curr_cell_fea = np.zeros((features_configs[f]['n_top_conf'][1]-features_configs[f]['n_top_conf'][0]))
                if len(curr_cell)>0:
                    curr_cell = curr_cell[np.argsort(curr_cell[:,4])[::-1]]
                    curr_cell = np.round(curr_cell[features_configs[f]['n_top_conf'][0]:features_configs[f]['n_top_conf'][1]],decimals=3)        
                    curr_cell_fea[:len(curr_cell)]
                tmp_features.extend(curr_cell_fea)
                features_names.extend([f'{c}_top_conf_{i}' for i in range(features_configs[f]['n_top_conf'][0],features_configs[f]['n_top_conf'][1])])
        if features_configs[f]['type'] == 'hist':
            fea = get_hist(result, cell_cls_list,
                           thresh=features_configs[f]['thresh'],
                           top_num_all_class=features_configs[f]['top_num_all_class'],
                           top_num_each_class=features_configs[f]['top_num_each_class'],
                           normalize_axis=features_configs[f]['normalize_axis'],
                           conf_bins=features_configs[f]['conf_bins'],
                           conf_range=features_configs[f]['conf_range'],cumsum=features_configs[f]['cumsum'])
            tmp_features.extend(fea.reshape(-1,))
            
            features_names.extend(get_feature_names(prefix=f,cell_type=cell_cls_list,shape=fea.shape).reshape(-1,))
            
    return tmp_features, features_names