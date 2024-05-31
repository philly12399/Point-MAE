import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
import open3d as o3d
import math
import pickle
from datetime import datetime
@DATASETS.register_module()
class Kitti_gtdet(data.Dataset):
    def __init__(self, config):
        self.info_path = config.INFO_PATH
        self.pcd_root = config.PCD_PATH
        self.seq = sorted(config.SEQ)
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.additional_cfg = config.additional_cfg
        self.data_info = {}
        tmp_data_info = IO.get(self.info_path)
        for s in self.seq:
            assert s in tmp_data_info
            self.data_info[s] = tmp_data_info[s]  

        self.file_list = []
        self.data_info_list = []
        for s in self.seq:
            for info in self.data_info[s]:
                ## additional pointmae info 
                info['seq'] = s
                outpath = f'{info["seq"]}/{info["velodyne_idx"]}_{info["obj_det_idx"]}_{info["obj"]["obj_type"]}'
                info['mae_vis_path'] = outpath
                info['mae_dense_path'] = outpath+".bin"
                info['valid'] = info['num_points_in_gt']>=self.additional_cfg.MIN_POINTS
                self.data_info_list.append(info)
                self.file_list.append(os.path.join(self.pcd_root, s,info['path']))
                
        ##output info
        info_output = os.path.join(config.additional_cfg.TARGET_PATH, "info.pkl")
        if  os.path.exists(info_output):
            now = datetime.now()
            nowd = now.strftime("-%m-%d-%H-%M")
            info_output = os.path.join(config.additional_cfg.TARGET_PATH, f"info{nowd}.pkl")
            print(f'info.pkl exist, write info to info{nowd}.pkl')
                        
        with open(info_output, 'wb') as file:
            pickle.dump(self.data_info, file)
        
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')
        self.permutation = np.arange(self.npoints)

    def __getitem__(self, idx):

        info = self.data_info_list[idx]
        box = info['obj']['box3d']
        data = IO.get(self.file_list[idx]).astype(np.float32)    
        if(self.additional_cfg.REFLECT_AUG):   
            data = reflect_augmentation(data, box)
            if(self.additional_cfg.ALIGN_XY):
                R_inv = rotz(-box['roty']) 
                data = np.dot(R_inv, data.T).T 
        # empty_voxel = get_voxel(data, bbox, self.additional_cfg.VOXEL_SIZE)
        # empty_voxel = torch.from_numpy(empty_voxel).float()
        empty_voxel = torch.from_numpy(np.array([0,0,0])).float()
        # data, centroid, m  = self.pc_norm(data)
        if(len(data) == 0):
            data = np.array([[0,0,0]])            
        data = torch.from_numpy(data).float()
        outpath = info['mae_vis_path']
        return data, outpath, empty_voxel, 
    
    def __len__(self):
        return len(self.file_list)
    
def reflect_augmentation(pcd, box):
    R = rotz(box['roty']) 
    R_inv =  rotz(-box['roty']) 
    # 旋轉到和xy對齊
    pcd = np.dot(R_inv, pcd.T).T        
    # 對長軸鏡射，使pcd左右方都有
    if(box['l'] > box['w']):
        Reflect = np.array([[1,0,0],[0,-1,0],[0,0, 1]])
    else :
        Reflect = np.array([[-1,0,0],[0,1,0],[0,0, 1]])
    pcdR = np.dot(Reflect, pcd.T).T
    pcd = np.concatenate((pcd, pcdR), axis=0)
    # 轉回來
    pcd = np.dot(R, pcd.T).T
    return pcd
    
def get_voxel(pcd, box, voxel_size = 0.3):
    R = rotz(box['roty']) 
    R_inv =  rotz(-box['roty']) 
    # 旋轉到和xy對齊        
    pcd = np.dot(R_inv, pcd.T).T
    # voxelize
    voxel, empty_voxel = voxelize(pcd, box, voxel_size)
    # 轉回來
    pcd = np.dot(R, pcd.T).T
    empty_voxel = np.dot(R, empty_voxel.T).T
    return empty_voxel
    
def voxelize(pcd, box, voxel_size=0.3):
    l, w, h = box['l'], box['w'], box['h']
    ln,wn,hn = int(l/voxel_size),int(w/voxel_size),int(h/voxel_size)
    ln,wn,hn = math.ceil(l/voxel_size),math.ceil(w/voxel_size),math.ceil(h/voxel_size)
    
    voxel = []    
    for i in range(0, ln):
        for j in range(0, wn):
            for k in range(0, hn):
                voxel.append({
                    'x': i * voxel_size - l/2 + voxel_size/2 ,
                    'y': j * voxel_size - w/2 + voxel_size/2 ,
                    'z': k * voxel_size - h/2 + voxel_size/2 ,
                    'l': voxel_size,
                    'w': voxel_size,
                    'h': voxel_size,
                    'cnt':0
                })

    pmax = pcd.max(0)-voxel_size/2
    pmin = pcd.min(0)+voxel_size/2            
    for p in pcd:
        i,j,k = int((p[0]+l/2)/voxel_size), int((p[1]+w/2)/voxel_size), int((p[2]+h/2)/voxel_size)
        idx = i*wn*hn+j*hn+k
        if(idx<len(voxel)):
            voxel[idx]['cnt']+=1
    
        
    empty=[]
    for v in voxel:
        if(v['cnt'] == 0):  
            if(in_range(np.array([v['x'],v['y'],v['z']]), pmax, pmin)):       
                empty.append([v['x'], v['y'],v['z']])
    empty = np.array(empty) 
    return voxel,empty
    
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
def in_range(v, pmax, pmin):
    return (v<=pmax).all() and (v>=pmin).all()