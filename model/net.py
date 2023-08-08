from torch import nn
import torch
from src.math import r6d_to_rotation_matrix


def nn_block(input,hidden,output):
    return nn.Sequential(nn.Linear(input, hidden),
                            nn.LeakyReLU(),
                            nn.Linear(hidden, output))

class bm(nn.Module):
    def __init__(self):
        super(bm, self).__init__()
        self.gender_embed = nn.Embedding(2,3)
        self.body_inference = self.nn_block(6, 1024, 19*3)

    def nn_block(self, input, hidden, output):
        return nn.Sequential(nn.Linear(input, hidden),
                             nn.Dropout(0.2),
                             nn.LeakyReLU(),
                             nn.Linear(hidden, hidden),
                             nn.LeakyReLU(),
                             nn.Linear(hidden, output))
    def forward(self,body_parms):
        gender = self.gender_embed(body_parms[:,0].to(torch.int))
        body = torch.cat([gender,body_parms[:,1:]], dim=1)
        body = self.body_inference(body)
        return body

class sub_pose(nn.Module):
    def __init__(self,input,output):
        super(sub_pose,self).__init__()
        self.parms_embed = nn_block(6, 64,128)
        self.data_embed = nn_block(input,64,128)
        self.bilstm = nn.LSTM(input_size=128,hidden_size=128,num_layers=2,batch_first=True,bidirectional=False)
        self.decode = nn_block(128,64,output)

    def forward(self, data, hc=None):
        b = data.size(0)
        s = data.size(1)
        data = data.reshape(b * s, data.size(2))
        data_embed = self.data_embed(data)
        data_embed = data_embed.reshape(b, s, data_embed.size(-1))
        y, hc = self.bilstm(data_embed, hc)
        y = y.reshape(b * s, y.size(-1))
        y = self.decode(y)
        y = y.reshape(b, s, y.size(-1))
        return y, hc

class integ(nn.Module):
    def nn_block(self,input,hidden,output):
        return nn.Sequential(nn.Linear(input, hidden),
                                    nn.LeakyReLU(),
                             nn.Linear(hidden, hidden),
                             nn.LeakyReLU(),
                                    nn.Linear(hidden, output))
    def __init__(self):
        super(integ, self).__init__()
        self.mlp = self.nn_block(12,72,256)
        self.lstm = nn.LSTM(input_size=256,hidden_size=256,num_layers=2,batch_first=True)
        self.mlp2 = self.nn_block(256,72,3)

    def forward(self,input,hc):
        b = input.size(0)
        s = input.size(1)
        j = input.size(2)
        x = self.mlp(input.view(b * s, j))
        x, hc = self.lstm(x.view(b, s, x.size(1)),hc)
        x = self.mlp2(x.reshape(b * s, x.size(2)))
        x = x.view(b, s, x.size(1))
        return x,hc

class kinect_chain(nn.Module):
    def __init__(self):
        super(kinect_chain, self).__init__()

        self.bm = bm()
        self.sub0 = nn_block(3 * 9 + 10 * 3 + 10 * 3, 128, 2 * 6)
        self.sub1 = nn_block(3*9+10*3+10*3+2*6,128,3*6)
        self.sub2 = nn_block(3*9+10*3+10*3+5*6,128,3*6)
        self.sub3 = nn_block(3*9+10*3+10*3+8*6,128,6)
        self.sub4 = nn_block(5*9+9*3+9*3+9*6,128,3*6)
        self.sub5 = nn_block(5*9+9*3+9*3+12*6,128,3*6)

    def forward(self,parms,joints,ori):
        #     parms--[b,4]
        #     joints--[b,19*3]
        #     ori--[b,5*4]
        body = self.bm(parms)
        x = torch.cat([body[:,9*3:],joints[:,9*3:],ori[:,:3*9]],dim=1)
        x0 = self.sub0(x)
        x1 = self.sub1(torch.cat([x,x0],dim=1))
        x2 = self.sub2(torch.cat([x,x0,x1],dim=1))
        x3 = self.sub3(torch.cat([x,x0,x1,x2],dim=1))
        x = torch.cat([body[:,:9*3],joints[:,:9*3],ori],dim=1)
        x4 = self.sub4(torch.cat([x,x0,x1,x2,x3],dim=1))
        x5 = self.sub5(torch.cat([x,x0,x1,x2,x3,x4],dim=1))
        return torch.cat([x5,x4,x3,x2,x1,x0],dim=1),body.reshape(body.size(0),19,3)

class FIP(nn.Module):
    def __init__(self):
        super(FIP, self).__init__()
        self.spinepose = sub_pose(36+128,13*3)
        self.hippose = sub_pose(24+128,6*3)
        self.integ = integ()
        self.ikp = kinect_chain()
        self.gender_embed = nn.Embedding(2,3)
        self.ini_pose_mlp = nn.Sequential(nn.Linear(15*9+19*3, 128),
                             nn.Dropout(0.2),
                             nn.LeakyReLU(),
                             nn.Linear(128, 128),
                             nn.LeakyReLU(),
                             nn.Linear(128, 5*3))

    def reset(self,pose_ini,parms):
        self.pose_ini = pose_ini[:, :15, ...]
        self.parms = parms
        self.pre_body = self.ikp.bm(parms).reshape(1, 19, 3)
        self.leaf_position = self.ini_pose_mlp(torch.cat([self.pre_body.view(1, 19 * 3), pose_ini.view(1, 15 * 9)], dim=-1))


    def forward_online(self,acc,ori,integ_hc=None,hip_hc=None,spine_hc = None):
        b = acc.size(0)
        s = acc.size(1)
        # obtain the relative sensor output
        # the input ori is derived by root
        parms = self.parms
        ori = ori[..., :45]
        acc = acc[..., :15].reshape(b, s, 5, 3)

        leaf_position = self.leaf_position
        leaf_position = leaf_position.reshape(b, 5, 1, 3)

        acc_q = torch.cat([acc.transpose(1, 2).reshape(b * 5, s, 3),  ori.view(b,s,5,9).transpose(1, 2).reshape(b * 5, s, 9)], dim=-1)
        integ_pose, integ_hc = self.integ(acc_q, integ_hc)

        integ_pose = (leaf_position + integ_pose.reshape(b, 5, s, 3)).transpose(1, 2)
        leaf_position = integ_pose

        gender = self.gender_embed(parms[:, 0].to(torch.int))
        parms_sp = torch.cat([gender, parms[:, 1:]], dim=1)
        spine_bm = self.spinepose.parms_embed(parms_sp).unsqueeze(1).repeat(1, s, 1)
        spine = torch.cat([integ_pose[:, :, :3].reshape(b, s, 9), ori[:, :, :3 * 9], spine_bm], dim=2)

        gender = self.gender_embed(parms[:, 0].to(torch.int))
        parms_hip = torch.cat([gender, parms[:, 1:]], dim=1)
        hip_bm = self.hippose.parms_embed(parms_hip).unsqueeze(1).repeat(1, s, 1)
        hip = torch.cat([integ_pose[:, :, 3:].reshape(b, s, 6), ori[:, :, 3 * 9:], hip_bm], dim=2)

        spine, spine_hc = self.spinepose(spine,spine_hc)
        hip, hip_hc = self.hippose(hip,hip_hc)

        all_j = torch.cat(
            [hip[:, :, :2 * 3], spine[:, :, :3], hip[:, :, 2 * 3:4 * 3], spine[:, :, 3:6], hip[:, :, 4 * 3:],
             spine[:, :, 6:]], dim=-1)
        all_j = all_j.view(b * s, 19 * 3)
        ori = ori.view(b * s, 5 * 9)
        parms = parms.unsqueeze(1).repeat([1, s, 1]).reshape(b * s, 4)
        rot, _ = self.ikp(parms, all_j, ori)
        rot = r6d_to_rotation_matrix(rot).view(b * s, 15, 3, 3)
        return rot.view(b, s, 15 * 9), rot[:, :, :, :2].reshape([b, s, 15 * 6]), \
               all_j.reshape(b, s, 19 * 3), leaf_position.reshape(b, s, 5 * 3),integ_hc,spine_hc,hip_hc
