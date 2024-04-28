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

    
    if("additional_cfg" not in config):
        config.additional_cfg = EasyDict({
                'TARGET_PATH': './vis',
                'VIS_NUM': 10,
                'START_INDEX': 0,
                'REFLECT_AUG': True,
                'ALIGN_XY': True,
            })
    os.system(f"mkdir -p {config.additional_cfg.TARGET_PATH}")
    os.system(f'cp {args.config} {config.additional_cfg.TARGET_PATH}')
    config.dataset.test._base_.additional_cfg = config.additional_cfg
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    
    #  DDP
    if args.distributed:
        raise NotImplementedError()
    #config setting and copy
        
    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger = None):
    base_model.eval()  # set model to eval mode
    
    # useful_cate = [
    #     "02691156", #plane
    #     "04379243",  #table
    #     "03790512", #motorbike
    #     "03948459", #pistol
    #     "03642806", #laptop
    #     "03467517",     #guitar
    #     "03261776", #earphone
    #     "03001627", #chair
    #     "02958343", #car
    #     "04090263", #rifle
    #     "03759954", # microphone
    # ]
    useful_cate = [
        "02958343", #car
        "03790512", #motorbike
        "02924116", #bus
    ]
    iterator = iter(test_dataloader)
    for _ in range(config.additional_cfg.START_INDEX):
        next(iterator)
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data, empty_center) in enumerate(tqdm(iterator), start = config.additional_cfg.START_INDEX):
            # import pdb; pdb.set_trace()
            if  config.additional_cfg.VIS_NUM > 0 and idx > config.additional_cfg.VIS_NUM :
                break

            if  taxonomy_ids[0] not in useful_cate:
                continue
            a, b = 0, 0

            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'Wayside':
                points = data.cuda()    
                masked_center= None     
                if base_model.MAE_encoder.mask_type == 'voxel':                   
                    empty_center = empty_center.cuda()
                    masked_center = misc.fps(empty_center, 38)
                
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')
            # dense_points, vis_points = base_model(points, vis=True)
            dense_points, vis_points, centers, mask = base_model(points, masked_center,vis=True)
                            
            final_image = []
            data_path = f'{config.additional_cfg.TARGET_PATH}/vis/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points,a,b)
            final_image.append(points[150:650,150:675,:])

            vis_points = vis_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            vis_points = misc.get_ptcloud_img(vis_points,a,b)
            final_image.append(vis_points[150:650,150:675,:])

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points,a,b)
            final_image.append(dense_points[150:650,150:675,:])

            centers = centers.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
            centers = misc.get_ptcloud_img(centers,a,b)
            final_image.append(centers[150:650,150:675,:])
            
            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)
            if(masked_center is not None):
                masked_center =  masked_center.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path,'voxelmask.txt'),  masked_center, delimiter=';')
            mask =  mask.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'mask.txt'),  mask, delimiter=';')
        return
