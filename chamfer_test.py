import os
import sys
import torch
import unittest
import pdb
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.chamfer_dist import ChamferFunction

def test(track_path='./data/output_bytrackid/car_all'):
    track = {}
    dirlist=sorted(os.listdir(track_path))
    for i in range(len(dirlist)):
        path1 = os.path.join(track_path, dirlist[i])
        if(not os.path.isdir(path1)):
            dirlist.pop(i) 
    for t in dirlist:
        track[t] = []
        tpath = os.path.join(track_path, t)
        frame_num = (int(sorted(os.listdir(tpath))[-1][:6])+1)  
        for f in range(frame_num):            
            pcd_path=os.path.join(tpath, str(f).zfill(6)+'_dense_points.txt')
            track[t].append(pcd_path)
    
        
    ##取出每個track最具代表性的frame(chamfer最低)
    # rep=[]
    # for ti in track:
    #     if(int(ti)>10):
    #         break
    #     st = track[ti]
    #     l=[]  
    #     dis_between_frame=[]  
    #     for t1 in st:
    #         l.append(read_vis_points(t1))

    #     tensor_all=torch.cat(l,0)
    #     for t1 in l:
    #         tensor_cur=t1.repeat(len(l),1,1)
    #         dis=Chamfer(tensor_cur.cuda(), tensor_all.cuda())
    #         dis_between_frame.append(float(torch.mean(dis).cpu().numpy()))
        
    #     rep.append(dis_between_frame.index(min(dis_between_frame)))
    #     dis_between_frame.append(tdis.cpu().numpy())
    
    rep=[19, 33, 19, 27, 43, 31, 26, 45, 26] 
    l=[]
    dis_between_track=[]
    for i,ti in enumerate(track):
        if(i>=len(rep)):
            break
        l.append(read_vis_points(track[ti][rep[i]]))
    tensor_all=torch.cat(l,0)
    for t1 in l:
        tensor_cur=t1.repeat(len(l),1,1)
        dis=Chamfer(tensor_cur.cuda(), tensor_all.cuda())
        dis_between_track.append(dis.cpu().numpy())



        

    
def read_vis_points(pcd_path):
    arr=[]
    with open(pcd_path, 'r') as file:
        data = file.readlines()
        for d in data:
            d = d.replace('\n','').split(';')
            arr.append([float(d[0]), float(d[1]), float(d[2])])
    arr = torch.tensor(arr).reshape(1, -1, 3)   
    return arr

def Chamfer(x, y):
    dist1, dist2 = ChamferFunction.apply(x, y)
    return torch.mean(dist1,dim=1) + torch.mean(dist2,dim=1)

if __name__ == '__main__':
    test()   
 
# s=torch.mean(dist1) + torch.mean(dist2)           