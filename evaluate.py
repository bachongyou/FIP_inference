import torch
from tqdm import tqdm
import os

import time
from src import math
from collections import OrderedDict
from human_body_prior.body_model.body_model import BodyModel
from src import functions as sf
from model.net import FIP
from src.eval_tools import *


body_parm = [[0,186,84.24332236,55.0763866],[0,178, 80.61995365, 52.70750976],
             [0,187, 84.69624344, 55.37249621],[0,170, 76.99658495, 50.33863291],
             [0,180, 81.52579583, 53.29972897],[100,172, 78.59580262, 51.98239681],[0,178, 80.61995365, 52.70750976],
             [0,180, 81.52579583, 53.29972897],[0,187, 84.69624344, 55.37249621],[0,181, 76.99658495, 50.33863291]]
body_parm = torch.tensor(body_parm)/100
Jtr_parent = [-1, -1, -1, 0, 1, 2, 3, 4, 5, 8, 8, 8, 9, 10, 11, 13, 14, 15, 16]
Jtr_id = [1,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,19,20,21,0]
rot_id = [1,2,3,4,5,6,9,12,13,14,15,16,17,18,19,0]


def evaluate_dip(data_dir,meta_data):
    data = torch.load(data_dir)
    if not hasattr(data, 'T_pose'):
        meta_data = torch.load(meta_data)
        bm_fname = os.path.join(path.support_dir, 'body_models/smplh/male/model.npz')
        dmpl_fname = os.path.join(path.support_dir, 'body_models/dmpls/male/model.npz')
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
            device)
        bm_fname = os.path.join(path.support_dir, 'body_models/smplh/female/model.npz')
        dmpl_fname = os.path.join(path.support_dir, 'body_models/dmpls/female/model.npz')
        bm_fm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
            device)
        data['T_pose'] = []
        for i in data['tokens']:
            with torch.no_grad():
                betas = torch.from_numpy(meta_data[f's{i}']).to(device)
                if i == 6:
                    body = bm_fm(betas=betas)
                else:
                    body = bm(betas=betas)
                data['T_pose'].append(body.Jtr[0,:22].cpu().numpy())
    xs = [(a, r, t) for a, r, t in zip(data['acc'], data['ori'], data['tokens'])]
    local_mat = []
    ys = [(math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3), t,j[:,Jtr_id]) for p, t, j in zip(data['pose'], data['tran'], data['joints'])]
    offline_errs, online_errs = [], []
    for id_, xy in tqdm(enumerate(list(zip(xs, ys)))):
        x, y = xy
        T_pose = torch.tensor(data['T_pose'][id_][[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] -
                              data['T_pose'][id_][0])
        T_pose = torch.cat([T_pose[:3], T_pose[3:] - T_pose[Jtr_parent[3:]]])
        acc, ori, token = x
        # get root_based imu acc data
        acc = ori[:, 5:].transpose(-1, -2) @ (acc - acc[:, 5:]).unsqueeze(-1)
        acc = acc.squeeze(-1)
        # get root_based imu gyr data
        b = acc.size(0)
        ori[:, :5] = ori[:, 5:].transpose(-1, -2) @ ori[:, :5]

        acc, ori = acc.reshape(-1,18).to(device)/30, ori.flatten(1).to(device)

        body_p = body_parm[token-1].unsqueeze(0).to(device)
        pose_t, _, joints_t = y
        pose_t_global = get_global_pose(pose_t)[:,rot_id[:-1]]
        pose_ini = pose_t_global[0].unsqueeze(0).to(device)
        # get root_based joints label
        pose_p, joints_p, _,_,_ = model(body_p,acc,ori,pose_ini)
        pose_p = pose_p.reshape(b,15,3,3).cpu()
        pose = sf.glb2local(pose_p)
        local_mat.append(pose)
        err = eval_with_points(pose_p, pose_t_global, T_pose)
        online_errs.append(err)

    print('============== online ================')
    err_print(torch.stack(online_errs).mean(dim=0))



def evaluate_dip_online(data_dir,meta_data):
    global model
    data = torch.load(data_dir)
    if not hasattr(data, 'T_pose'):
        meta_data = torch.load(meta_data)
        bm_fname = os.path.join(path.support_dir, 'body_models/smplh/male/model.npz')
        dmpl_fname = os.path.join(path.support_dir, 'body_models/dmpls/male/model.npz')
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
            device)
        bm_fname = os.path.join(path.support_dir, 'body_models/smplh/female/model.npz')
        dmpl_fname = os.path.join(path.support_dir, 'body_models/dmpls/female/model.npz')
        bm_fm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
            device)
        data['T_pose'] = []
        for i in data['tokens']:
            with torch.no_grad():
                betas = torch.from_numpy(meta_data[f's{i}']).to(device)
                if i == 6:
                    body = bm_fm(betas=betas)
                else:
                    body = bm(betas=betas)
                data['T_pose'].append(body.Jtr[0,:22].cpu().numpy())

    xs = [(a, r, t) for a, r, t in zip(data['acc'], data['ori'], data['tokens'])]
    ys = [(math.axis_angle_to_rotation_matrix(p).view(-1, 24, 3, 3), t,j[:,Jtr_id]) for p, t, j in zip(data['pose'], data['tran'], data['joints'])]
    online_errs = []
    tss = []
    lats = []
    poses_p_glb= []
    poses_t_glb = []
    for id_, xy in tqdm(enumerate(list(zip(xs, ys)))):
        x, y = xy
        T_pose = torch.tensor(data['T_pose'][id_][[1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]] -
                              data['T_pose'][id_][0])
        T_pose = torch.cat([T_pose[:3], T_pose[3:] - T_pose[Jtr_parent[3:]]])
        acc, ori, token = x
        # get root_based imu acc data
        acc = ori[:, 5:].transpose(-1, -2) @ (acc - acc[:, 5:]).unsqueeze(-1)
        acc = acc.squeeze(-1)
        ori[:, :5] = ori[:, 5:].transpose(-1, -2) @ ori[:, :5]
        acc, ori = acc.reshape(-1,18).to(device)/30, ori.flatten(1).to(device)
        body_p = body_parm[token-1].unsqueeze(0).to(device)
        pose_t, _, joints_t = y
        pose_t_global = get_global_pose(pose_t)[:,rot_id[:-1]]
        pose_ini = pose_t_global[0].unsqueeze(0).to(device)
        model.reset(pose_ini,body_p)
        poses_p = []
        integ_hc = None
        hip_hc = None
        spine_hc = None
        t = time.time()
        for i in range(len(acc)):
            acc_ = acc[i:i+1].unsqueeze(0)
            ori_ = ori[i:i+1].unsqueeze(0)
            s = time.time()
            rot, _,_, _, integ_hc, spine_hc, hip_hc= model.forward_online(acc_,ori_,integ_hc=integ_hc,hip_hc=hip_hc,spine_hc = spine_hc)
            # rot, _,_,_, _, integ_hc, spine_hc, hip_hc,_,_= model.forward_online(acc_,ori_,integ_hc=integ_hc,hip_hc=hip_hc,spine_hc = spine_hc,with_root=False)
            lats.append(time.time()-s)
            poses_p.append(rot.squeeze())
        gap = time.time()-t
        tss.append(gap/i)
        pose_p = torch.stack(poses_p).reshape(-1,15,3,3).cpu()
        online_errs.append(eval_with_points(pose_p, pose_t_global, T_pose))
        poses_p_glb.append(pose_p)
        poses_t_glb.append(pose_t_global)

    print('============== evaluating ================')
    del model
    err_print(torch.stack(online_errs).mean(dim=0))
    eval_mesh_Aang(poses_p_glb,poses_t_glb,path.smpl_file)
    print('average FPS: %.2f fps' % float(1/torch.mean(torch.tensor(tss)).item()))
    print('average latency: %.2f ms'% float(1000*torch.mean(torch.tensor(lats)).item()))




if __name__ == '__main__':
    class config:
        dipimu_dir = 'data/imu_test.pt'
        dipimu_betas = 'data/dip_betas.pt'
        support_dir = 'data/support_data'
        smpl_file = 'data/SMPL_male.pkl'
        # model_dir = "ckpt/best_2.pt"
        model_dir = "ckpt/best_model.pt"

    path = config
    losses = OrderedDict()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FIP()

    model.load_state_dict(torch.load(path.model_dir))
    model.to(device)
    model.eval()
    with torch.no_grad():
        evaluate_dip_online(path.dipimu_dir,path.dipimu_betas)




