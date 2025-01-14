import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from tqdm import tqdm
import cv2
import numpy as np
import yaml
from easydict import EasyDict

def test_net(args, config):
    
    ## config insert to dataset
    if("additional_cfg" not in config):
        config.additional_cfg = EasyDict({
                'TARGET_PATH': './vis',
                'VIS_NUM': 10,
                'START_INDEX': 0,
                'REFLECT_AUG': True,
                'ALIGN_XY': True,
                'CONF_THRES':0.0,
            })
    if("save_vis_txt" not in config):
        config.save_vis_txt=True
        
    config.additional_cfg.MIN_POINTS = config.model.group_size
    config.dataset.test._base_.additional_cfg = config.additional_cfg
    
    ##makedir and copy config
    os.system(f"mkdir -p {config.additional_cfg.TARGET_PATH}")
    os.system(f'cp {args.config} {config.additional_cfg.TARGET_PATH}')
    for s in sorted(config.dataset.test._base_.SEQ):
        if(config.save_vis_txt):
            os.system(f"mkdir -p {config.additional_cfg.TARGET_PATH}/vis/{s}")
        os.system(f"mkdir -p {config.additional_cfg.TARGET_PATH}/gt_database/{s}")
        
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    ## build dataset/dataloader
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    
    
    
    ## build model and start inference
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)      
    test(base_model, test_dataloader, args, config, logger=logger)

# visualization
def test(base_model, test_dataloader, args, config, logger = None):
    base_model.eval()  # set model to eval mode
    MIN_POINTS = config.model.group_size
    iterator = iter(test_dataloader)
    for _ in range(config.additional_cfg.START_INDEX):
        next(iterator)
    logger = os.path.join(config.additional_cfg.TARGET_PATH,'log.txt')
    with torch.no_grad():
        for idx, (data, info_out, empty_center, valid) in enumerate(tqdm(iterator), start = config.additional_cfg.START_INDEX):
            # import pdb; pdb.set_trace()
            if  config.additional_cfg.VIS_NUM > 0 and idx - config.additional_cfg.START_INDEX > config.additional_cfg.VIS_NUM :
                break
            a, b = 0, 0
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'AOT':
                points = data.cuda()    
                masked_center= None     
                if base_model.MAE_encoder.mask_type == 'voxel':                   
                    empty_center = empty_center.cuda()
                    masked_center = misc.fps(empty_center, 38)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            # dense_points, vis_points = base_model(points, vis=True)                        
            final_image = []
            if(valid[0]==True):
                dense_points, vis_points, centers, mask = base_model(points, masked_center,vis=True)
                if(config.save_vis_txt):
                    out_path_vis = os.path.join(f"{config.additional_cfg.TARGET_PATH}/vis/",info_out[0])
                    if not os.path.exists(out_path_vis):
                        os.makedirs(out_path_vis)     
                    save_points_and_img(points, os.path.join(out_path_vis, 'gt.txt'), a, b, final_image)
                    save_points_and_img(vis_points, os.path.join(out_path_vis, 'vis.txt'), a, b, final_image)
                    save_points_and_img(dense_points, os.path.join(out_path_vis, 'dense_points.txt'), a, b, final_image)
                    save_points_and_img(centers, os.path.join(out_path_vis, 'center.txt'), a, b, final_image)                        
                    if(masked_center is not None):
                        save_points_and_img(masked_center, os.path.join(out_path_vis, 'voxelmask.txt'), a, b)
                    save_points_and_img(mask, os.path.join(out_path_vis, 'vmask.txt'), a, b)
                
                    img = np.concatenate(final_image, axis=1)
                    img_path = os.path.join(out_path_vis, f'plot.jpg')
                    cv2.imwrite(img_path, img)
                
                out_path_dense = os.path.join(f"{config.additional_cfg.TARGET_PATH}/gt_database/")
                save_points_to_bin(dense_points,out_path_dense+info_out[0]+".bin") 
            
                with open(logger, 'a') as f:
                    f.write(f'{idx} {info_out[0]}\n')    
        return
    
def save_points_and_img(points, path, a, b, img=None):
    points = points.squeeze(axis=0).detach().cpu().numpy()    
    np.savetxt(path, points, delimiter=';')
    if(img is not None):
        points = misc.get_ptcloud_img(points,a,b)        
        img.append(points[150:650,150:675,:])

def save_points_to_bin(points, path):
    points = points.squeeze(axis=0).detach().cpu().numpy()
    zeros_column = np.zeros((points.shape[0], 1))
    points = np.concatenate((points, zeros_column), axis=1).astype(np.float32)
    points.tofile(path)
    
