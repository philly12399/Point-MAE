import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
import open3d as o3d
import math
@DATASETS.register_module()
class Wayside(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        
        self.voxel_size = config.VOXEL_SIZE
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.data_info = IO.get(os.path.join(self.data_root, f'{self.subset}.pkl'))['0004']

        self.sample_points_num = config.npoints
        self.whole = config.get('whole')
        
        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        info = self.data_info[idx]
        bbox = info['obj']['box3d']
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)        
        data = reflect_augmentation(data, bbox)
        # data, centroid, m  = self.pc_norm(data)
        
        empty_voxel = get_voxel(data, bbox, self.voxel_size)
        empty_voxel = torch.from_numpy(empty_voxel).float()
        
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data, empty_voxel
    
    def __len__(self):
        return len(self.file_list)
    
def reflect_augmentation( pcd, box):
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
    for p in pcd:
        i,j,k = int((p[0]+l/2)/voxel_size), int((p[1]+w/2)/voxel_size), int((p[2]+h/2)/voxel_size)
        idx = i*wn*hn+j*hn+k
        voxel[idx]['cnt']+=1
    
        
    empty=[]
    for v in voxel:
        if(v['cnt'] == 0):
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
    