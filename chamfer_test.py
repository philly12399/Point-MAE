import os
import sys
import torch
import unittest
import pdb
import pickle
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.chamfer_dist import ChamferFunction
from scipy.spatial.distance import directed_hausdorff
def test(track_path='./data/output_bytrackid/car_all'):
    track = {}
    dirlist=sorted(os.listdir(track_path))
    for i in range(len(dirlist)):
        path1 = os.path.join(track_path, dirlist[i])
        if(not os.path.isdir(path1)):
            pkl=dirlist.pop(i) 
    with open(os.path.join(track_path, pkl), 'rb') as file:
        info = pickle.load(file)
    # print(info['000002'][19])
    for t in dirlist:
        track[t] = []
        tpath = os.path.join(track_path, t)
        frame_num = (int(sorted(os.listdir(tpath))[-1][:6])+1)  
        for f in range(frame_num):            
            pcd_path=os.path.join(tpath, str(f).zfill(6)+'_dense_points.txt')
            track[t].append(pcd_path)
            
   


    

    rep=[]
    for tid in track:
        frames = track[tid]
        l=[]  
        m1=[] 
        m2=[]
        for f in frames:
            l.append(read_vis_points(f))
        # tensor_all=torch.cat(l,0)
        for f1 in l:
            chamfer=[]
            haus = []
            for f2 in l:
                chamfer_dis=FilterChamfer(f1, f2)
                haus_dis = FilterHausdorf(f1,f2)
                chamfer.append(chamfer_dis)
                haus.append(haus_dis)
            m1.append(chamfer)
            m2.append(haus)
        break
    pdb.set_trace()
        # rep.append(dis_between_frame.index(min(dis_between_frame)))
        # dis_between_frame.append(dis.cpu().numpy())
    
    # rep=[19, 33, 19, 27, 43, 31, 26, 45, 26] 
    # l=[]
    # dis_between_track=[]
    # for i,ti in enumerate(track):
    #     if(i>=len(rep)):
    #         break
    #     l.append(read_vis_points(track[ti][rep[i]]))
    # tensor_all=torch.cat(l,0)
    # for t1 in l:
    #     tensor_cur=t1.repeat(len(l),1,1)
    #     dis=Chamfer(tensor_cur.cuda(), tensor_all.cuda())
    #     dis_between_track.append(dis.cpu().numpy())



        

    
def read_vis_points(pcd_path):
    arr=[]
    with open(pcd_path, 'r') as file:
        data = file.readlines()
        for d in data:
            d = d.replace('\n','').split(';')
            arr.append([float(d[0]), float(d[1]), float(d[2])])
    arr = torch.tensor(arr).reshape(1, -1, 3)   
    return arr

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
    return float(DirectChamfer(a.cuda(),b_filtered.cuda()).cpu())

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
    haus = DirectHausdorf(a,b_filtered)
    # draw([a,b,b_filtered])
    return haus
    

import open3d as o3d
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
    
if __name__ == '__main__':
    test()   
 
# s=torch.mean(dist1) + torch.mean(dist2)           