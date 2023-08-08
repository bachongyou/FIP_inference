import torch
from . import math, kinematic_model

rot_id = [1,2,3,4,5,6,9,12,13,14,15,16,17,18,19,0]
Jtr_parent = [-1, -1, -1, 0, 1, 2, 3, 4, 5, 8, 8, 8, 9, 10, 11, 13, 14, 15, 16]
Jtr_mask = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

def glb2local(glb_pose):
    global_full_pose = torch.eye(3, device=glb_pose.device).repeat(glb_pose.shape[0], 24, 1, 1)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
               16, 17, 18, 19]
    global_full_pose[:, [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]] = glb_pose
    for i in range(len(parents)-1 , 3 , -1):
        p = global_full_pose[:,parents[i]]
        global_full_pose[:,i] = p.transpose(-1,-2) @ global_full_pose[:,i]
    global_full_pose[:, [0, 7, 8, 10, 11, 20, 21,22,23]] = torch.eye(3, device=glb_pose.device)
    return global_full_pose

def get_global_pose(pose):
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
               16, 17, 18, 19]
    rot_smpl_root = pose[:, :22, :] * 1
    for i in range(4, len(parents)):
        rot_smpl_root[:, i, :] = rot_smpl_root[:, parents[i], :] @ rot_smpl_root[:, i, :]
    return rot_smpl_root

def err_print(errors):
    for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)', 'Jitter (km/s^3)'
                              ]):
        print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))

def eval_with_points(pose_p, pose_t,T_pose,fps=60,align_joint=0, joint_p=None, joint_t=None):
    f = fps
    def forward_kinematics(body, pose):
        """
        :param body: [19,3]
        :param pose: [b,15,3,3]
        :return: joints[b,19,3]
        """
        b = pose.size(0)
        body = body.unsqueeze(0).repeat(b, 1, 1)
        p_tree = [[-1, -1, -1], [0, 1, 2], [3, 4, 5], [6, 6, 6], [7, 8, 9], [11, 12], [13, 14]]
        k_tree = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16], [17, 18]]
        x1 = body[:, k_tree[0]]
        x2 = x1 + torch.bmm(pose[:, p_tree[1]].reshape(b * 3, 3, 3),
                            body[:, k_tree[1]].reshape(b * 3, 3, 1)).reshape(b, 3, 3)
        x3 = x2 + torch.bmm(pose[:, p_tree[2]].reshape(b * 3, 3, 3),
                            body[:, k_tree[2]].reshape(b * 3, 3, 1)).reshape(b, 3, 3)
        x4 = x3[:, [2, 2, 2]] + torch.bmm(pose[:, p_tree[3]].reshape(b * 3, 3, 3),
                                          body[:, k_tree[3]].reshape(b * 3, 3, 1)).reshape(b, 3, 3)
        x5 = x4 + torch.bmm(pose[:, p_tree[4]].reshape(b * 3, 3, 3),
                            body[:, k_tree[4]].reshape(b * 3, 3, 1)).reshape(b, 3, 3)
        x6 = x5[:, [1, 2]] + torch.bmm(pose[:, p_tree[5]].reshape(b * 2, 3, 3),
                                       body[:, k_tree[5]].reshape(b * 2, 3, 1)).reshape(b, 2, 3)
        x7 = x6 + torch.bmm(pose[:, p_tree[6]].reshape(b * 2, 3, 3),
                            body[:, k_tree[6]].reshape(b * 2, 3, 1)).reshape(b, 2, 3)
        return torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=1)
    if joint_p is None:
        joint_p = forward_kinematics(T_pose, pose_p)
    if joint_t is None:
        joint_t = forward_kinematics(T_pose, pose_t)

    offset_from_p_to_t = (joint_t[:, align_joint] - joint_p[:, align_joint]).unsqueeze(1)
    je = (joint_p + offset_from_p_to_t - joint_t).norm(dim=2)  # N, J
    gae = math.radian_to_degree(math.angle_between(pose_p, pose_t).view(pose_p.shape[0], -1))  # N, J
    mgae = gae[:, [0, 1, 11, 12]]
    jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2)  # N, J
    errs = torch.tensor([[mgae.mean(), mgae.std(dim=0).mean()],
                         [gae.mean(), gae.std(dim=0).mean()],
                         [je.mean()*100, je.std(dim=0).mean()*100],
                         [jkp.mean()/1000, jkp.std(dim=0).mean()/1000]
                         ])

    return errs

def eval_mesh_Aang(pose_glb,pose_gt_glb,smpl_file):
    pose_lc = [glb2local(i) for i in pose_glb]
    pose_gt_glb = [glb2local(i) for i in pose_gt_glb]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = kinematic_model.ParametricModel(smpl_file, device="cuda" if torch.cuda.is_available() else "cpu")
    mesh_errs = []
    aangs = []
    for x,y in zip(pose_lc,pose_gt_glb):
        x, y = x.to(device),y.to(device)
        x[:,[0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=x.device)
        y[:,[0, 7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=y.device)
        x_glb, _, mesh_p = m.forward_kinematics(x, calc_mesh=True)
        y_glb, _, mesh_t = m.forward_kinematics(y, calc_mesh=True)
        mesh_errs.append(torch.tensor([(mesh_p - mesh_t).norm(dim=2).mean().cpu(),(mesh_p - mesh_t).norm(dim=2).std(dim=0).mean().cpu()]))

        aang = math.radian_to_degree(math.angle_between(y_glb, x_glb).view(x_glb.shape[0], -1))
        aangs.append(torch.tensor([aang.mean(),aang.std(dim=0).mean().cpu()]))

    print("Mesh Error (cm): %.2f (+/- %.2f)\nAll angular Error (deg): %.2f (+/- %.2f)" %
          (torch.stack(mesh_errs).mean(dim=0)[0].item() * 100,
           torch.stack(mesh_errs).mean(dim=0)[1].item() * 100,
           torch.stack(aangs).mean(dim=0)[0].item(),
           torch.stack(aangs).mean(dim=0)[1].item()))
