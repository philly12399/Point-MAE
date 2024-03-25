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
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        data = self.reflect_augmentation(data, info['obj']['box3d'])
        # data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data, info 

    def reflect_augmentation(self, pcd, box):
        y=box['roty']
        R=np.array([[math.cos(y),math.sin(y),0],[-math.sin(y),math.cos(y),0],[0,0,1]])
        # R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0,box['roty'])) 
        # R_inv = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0,-box['roty']))
        R_inv=np.array([[math.cos(-y),math.sin(-y),0],[-math.sin(-y),math.cos(-y),0],[0,0,1]])
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
    
    def __len__(self):
        return len(self.file_list)