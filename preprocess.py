import pickle
from src import functions as sf
from tqdm import tqdm
import torch
import numpy as np
import os
from human_body_prior.body_model.body_model import BodyModel

support_dir = 'data/support_data'
subject_gender = 'male'
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bm_fname = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters
bm1 = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)

subject_gender = 'female'
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bm_fname = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm2 = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)

class paths:
    raw_dipimu_dir = './DIP_IMU_and_Others/DIP_IMU'  # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = './DIP_IMU_and_Others'  # output path for the preprocessed DIP-IMU dataset

    pred_beta_file = './dip_betas.pt'


def process_dipimu(mode='test'):
    imu_mask = [0, 7, 8, 11, 12, 2]
    if mode == "test":
        test_split = ['s_09', 's_10']
        save_dir = os.path.join(paths.dipimu_dir, 'imu_test.pt')
    else:
        test_split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
        save_dir = os.path.join(paths.dipimu_dir, 'imu_train.pt')
    pred_beta = torch.load(paths.pred_beta_file)
    accs, oris, poses, trans, joints, tokens = [], [], [], [], [], []

    for subject_name in tqdm(test_split):
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
                tokens.append(int(subject_name[-2:]))
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

            # joints
            if subject_name == "s_10":
                beta = pred_beta['s10']
            else:
                beta = pred_beta[f's{subject_name[-1]}']
            bm = bm2 if subject_name == 's_06' else bm1
            time_len = len(pose)

            body_parms = {
                'root_orient': torch.Tensor(pose[:, :3]).to(comp_device),  # controls the global root orientation
                'pose_body': torch.Tensor(pose[:, 3:66]).to(comp_device),  # controls the body
                'betas': torch.Tensor(np.repeat(beta, repeats=time_len, axis=0)).to(comp_device),
                # controls the body shape. Body shape is static
            }
            length = time_len // 400
            remain = time_len % 400

            for j in range(length):
                body = bm(**{k: v[j * 400:(j + 1) * 400] for k, v in body_parms.items() if
                             k in ['root_orient', 'pose_body', 'betas']})
                if j == 0:
                    joints_ = body.Jtr.cpu().numpy()[:, :22]
                else:
                    joints_ = np.r_[joints_, body.Jtr.cpu().numpy()[:, :22]]
            if remain > 0:
                body = bm(**{k: v[length * 400:] for k, v in body_parms.items() if
                             k in ['root_orient', 'pose_body', 'betas']})
                joints_ = np.r_[joints_, body.Jtr.cpu().numpy()[:, :22]]
            joints.append(joints_)

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans, 'joints': joints, 'tokens': tokens}, save_dir)
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)

def process_quat(data_dir,dipimu_quat_dir):
    data = torch.load(data_dir)
    quat = []
    for ori in tqdm(data['ori']):
        # get root_based imu acc data
        # # get root_based imu gyr data
        ori[:, :5] = ori[:, 5:].transpose(-1, -2) @ ori[:, :5]
        ori = ori.reshape(-1,3,3)
        result = []
        for mat in ori:
            q = sf.trace_method(mat)
            result.append(q)
        ori = torch.stack(result).reshape(-1,24)
        quat.append(ori)
    torch.save(quat, dipimu_quat_dir)

with torch.no_grad():
    process_dipimu('train')
    process_dipimu('test')