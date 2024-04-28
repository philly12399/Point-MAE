import os
import sys
import torch
import unittest
import pdb
import pickle
import numpy as np
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.chamfer_dist import ChamferFunction
from scipy.spatial.distance import directed_hausdorff
import open3d as o3d 

def test(track_path='./data/output_bytrackid/car_mark_all_rotxy'):
    track = {}
    dirlist=sorted(os.listdir(track_path))
    for i in range(len(dirlist)):
        path1 = os.path.join(track_path, dirlist[i])
        if(not os.path.isdir(path1)):
            pkl=dirlist.pop(i) 
    with open(os.path.join(track_path, pkl), 'rb') as file:
        info = pickle.load(file)
    vis = o3d.visualization.Visualizer()
    
    for t in dirlist:
        track[t] = []
        tpath = os.path.join(track_path, t)
        frame_num = (int(sorted(os.listdir(tpath))[-1][:6])+1)  
        for f in range(frame_num):            
            pcd_path=os.path.join(tpath, str(f).zfill(6)+'_dense_points.txt')
            track[t].append(pcd_path)
            break
        
    for i, tid in enumerate(track):
        t_info = info[tid]
        frames = track[tid]
        l=[]  
        for fi, f in enumerate(frames):
            pcd = read_vis_points(f)
            bbox = t_info[fi]['obj']['box3d']
            bbox['x'],bbox['y'],bbox['z'] = 0,0,0
            vis.create_window()
            # drawbox(vis,bbox)
            p = o3d.geometry.PointCloud()
            p.points = o3d.utility.Vector3dVector(pcd)
            vis.add_geometry(pcd)
            vis.run()
            vis.destroy_window()
            voxelize(pcd, bbox, 0.5)
            
            exit()
            # l.append()
            
        # tensor_all=torch.cat(l,0)
        for f1 in l:
            chamfer=[]
            haus = []
            for f2 in l:
                chamfer_dis = FilterChamfer(f1, f2)
                # haus_dis = FilterHausdorf(f1,f2)
                chamfer.append(chamfer_dis)
                # haus.append(haus_dis)
            avgm[0].append(sum(chamfer)/len(chamfer))
            
        avgmrk[0]=rank_list(avgm[0])
        rep.append(avgmrk[0].index(0))        
        rep_score[0].append(avgm[0][avgmrk[0].index(0)])
        # avgmrk[1]=rank_list(avgm[1])
    pdb.set_trace()
    # exit()
    
    

    
def voxelize(pcd, box, voxel_size=0.3):
    l, w, h = box['l'], box['w'], box['h']
    # ln,wn,hn = int(l/voxel_size),int(w/voxel_size),int(h/voxel_size)
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
        voxel[idx]['cnt']+=1
    empty=[]
    for v in voxel:
        if(v['cnt'] == 0):
            if(in_range(np.array([v['x'],v['y'],v['z']]), pmax, pmin)):
                empty.append(v)
    return voxel,empty

def read_vis_points(pcd_path):
    arr=[]
    with open(pcd_path, 'r') as file:
        data = file.readlines()
        for d in data:
            d = d.replace('\n','').split(';')
            arr.append([float(d[0]), float(d[1]), float(d[2])])
    
    # arr = torch.tensor(arr).reshape(1, -1, 3)   
    return np.array(arr)

def Chamfer(a, b):
    dist1, dist2 = ChamferFunction.apply(a, b)
    return (torch.mean(dist1,dim=1) + torch.mean(dist2,dim=1))/2

def DirectChamfer(a, b):
    dist1, dist2 = ChamferFunction.apply(a, b)
    return torch.mean(dist1,dim=1)
            
def FilterChamfer(a, b, dx = 0.1, dy = 0.1):
    a_range = [torch.min(a[:,:,0])-dx,torch.max(a[:,:,0])+dx, torch.min(a[:,:,1])-dy,torch.max(a[:,:,1])+dy]
    # yrange = [torch.min(y[:,:,0]),torch.max(y[:,:,0]), torch.min(y[:,:,1]),torch.max(y[:,:,1])]
    b_filtered = b[(b[:,:,0] >= a_range[0]) & (b[:,:,0] <= a_range[1]) & (b[:,:,1] >= a_range[2]) & (b[:,:,1] <= a_range[3])].reshape(1,-1,3)
    if(b_filtered.size()[1]<=0): #non overlap
        return 999.0
    # print(b.size(),'->',b_filtered.size())
    # draw([a.reshape(-1,3),b.reshape(-1,3),b_filtered.reshape(-1,3)])
    return float(Chamfer(a.cuda(),b.cuda()).cpu())
    # return float(Chamfer(a.cuda(),b_filtered.cuda()).cpu())

def Hausdorf(a, b):
    return directed_hausdorff(a, b)[0]/2 + directed_hausdorff(b,a)[0]/2

def DirectHausdorf(a, b):
    return directed_hausdorff(a, b)[0]

def FilterHausdorf(a,b,dx = 0.1, dy = 0.1):    
    a_range = [torch.min(a[:,:,0])-dx,torch.max(a[:,:,0])+dx, torch.min(a[:,:,1])-dy,torch.max(a[:,:,1])+dy]
    # yrange = [torch.min(y[:,:,0]),torch.max(y[:,:,0]), torch.min(y[:,:,1]),torch.max(y[:,:,1])]
    b_filtered = b[(b[:,:,0] >= a_range[0]) & (b[:,:,0] <= a_range[1]) & (b[:,:,1] >= a_range[2]) & (b[:,:,1] <= a_range[3])].reshape(-1,3)
    if(b_filtered.size()[0]<=0): #non overlap
        return 999.0
    a = a.reshape(-1,3)
    b = b.reshape(-1,3)
    haus = Hausdorf(a,b)
    # haus = Hausdorf(a,b_filtered)
    # draw([a,b,b_filtered])
    return haus

def draw(pcd) :
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    color=[[0,255,0],[0,0,0],[0,0,255],[255,0,0]] #G Black blue red 
    for i,p1 in enumerate(pcd):
        p= o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(p1)    
        p.paint_uniform_color(color[i])
        vis.add_geometry(p)
    
    vis.run()
    vis.destroy_window()
    
def drawbox(vis,box):
    b = o3d.geometry.OrientedBoundingBox()
    b.center = [box['x'],box['y'],box['z']]
    b.extent = [box['l'],box['w'],box['h']]
    vis.add_geometry(b)
    
def rank_list(input_list):
    ranked_list = sorted(range(len(input_list)), key=lambda x: input_list[x])
    return [ranked_list.index(i) for i in range(len(input_list))]

if __name__ == '__main__':
    test()   
 
# s=torch.mean(dist1) + torch.mean(dist2)           