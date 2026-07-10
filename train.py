import os
import glob
import json
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
from natsort import natsorted
import csv

import os
import glob
import warnings
import torch
import numpy as np
from torch.optim import Adam
import torch.utils.data as Data
from natsort import natsorted
import TransModels.LDMMorph as LDMMorph 

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config, default
from ldm.data.xcat_Motion_Seq import XCATSeqRegistration
from ldm.data.xcat_npz import XCATNPZRegistration
from omegaconf import OmegaConf
from torch.autograd import Variable
#用于xcat运动增强版本的训练
#原本是通过npz文件进行训练的，现在通过xcat_Motion.py进行训练
parser = ArgumentParser()
parser.add_argument("--resume", type=str,
                    dest="resume", default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-04-30T21-02-35_xcat-motion-ldm/checkpoints/last.ckpt',
                    help="pretrained model")
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=24001,
                    help="number of total iterations")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.4, 
                    help="smth_labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="data path for datasets (contains train/, val/, test/ subdirs)") 
parser.add_argument("--beta", type=float,
                    dest="beta",
                    default=0.8,
                    help="beta loss: range from 0.1 to 1.0")
parser.add_argument("--xcat", action="store_true",
                    dest="xcat",
                    help="Use XCAT dataset with motion augmentation (no npz required)")
parser.add_argument("--xcat_path", type=str,
                    dest="xcat_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="Data root for XCAT dataset")
parser.add_argument("--fixed_motion", action="store_true",
                    dest="fixed_motion",
                    help="Use XCAT fixed+motion dataset (fixed image + moving sequence frames)")
parser.add_argument("--fixed_motion_path", type=str,
                    dest="fixed_motion_path",
                    default='/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data',
                    help="Data root for fixed_motion dataset")
parser.add_argument("--loss_type", type=str, default='ncc',
                    choices=['ncc', 'mse'],
                    dest="loss_type",
                    help="Image domain loss type: ncc (default) or mse")
parser.add_argument("--no_ldm", action="store_true",
                    dest="no_ldm",
                    help="Disable LDM features, use learnable placeholders instead")
parser.add_argument("--ldm_config", type=str,
                    dest="ldm_config",
                    default=None,
                    help="LDM config file path")
opt = parser.parse_args()


lr = opt.lr
bs = opt.bs
iteration = opt.iteration
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
beta = opt.beta
t_enc = 1 

opt, unknown = parser.parse_known_args()
ckpt = None
if opt.ldm_config:
    configs = [opt.ldm_config]
else:
    # 默认使用非运动增强版本的LDM配置，如需更改请修改此处
    configs = ['/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/latent-diffusion/xcat_motion-ldm.yaml']
opt.ldm = configs
print(opt.resume)

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval() 
    return model

def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

def dice(pred1, truth1):
    if datapath=='acdc':
        VOI_lbls = [2,3]
    else:
        VOI_lbls = [1]
    dice_all=np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        
        dice_all[index]=intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_all)

def ncc_loss(fixed, moving, win_size=9):
    """Local Normalized Cross-Correlation loss - single pooling operation"""
    assert fixed.shape == moving.shape
    assert win_size % 2 == 1

    b, c, h, w = fixed.shape
    pad = win_size // 2

    fixed_pad = F.pad(fixed, [pad, pad, pad, pad], mode='reflect')
    moving_pad = F.pad(moving, [pad, pad, pad, pad], mode='reflect')

    patches_fix = fixed_pad.unfold(2, win_size, 1).unfold(3, win_size, 1)
    patches_mov = moving_pad.unfold(2, win_size, 1).unfold(3, win_size, 1)
    patches_fix = patches_fix.contiguous().view(b, c, h, w, -1)
    patches_mov = patches_mov.contiguous().view(b, c, h, w, -1)

    mean_fix = patches_fix.mean(dim=-1)
    mean_mov = patches_mov.mean(dim=-1)
    centered_fix = patches_fix - mean_fix.unsqueeze(-1)
    centered_mov = patches_mov - mean_mov.unsqueeze(-1)

    var_fix = (centered_fix ** 2).mean(dim=-1)
    var_mov = (centered_mov ** 2).mean(dim=-1)
    cross = (centered_fix * centered_mov).mean(dim=-1)

    eps = 1e-8
    ncc = cross / (torch.sqrt(var_fix.clamp(min=eps)) * torch.sqrt(var_mov.clamp(min=eps)) + eps)

    return 1.0 - ncc.mean()

def save_checkpoint(state, save_dir, save_filename, max_model_num=50):
    torch.save(state, save_dir + save_filename)
    # 只清理 .pth 文件，不影响可视化图片等其他文件
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*.pth')))
    
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*.pth')))

def train():
    global opt, datapath
    print(opt.resume)
    ckpt = opt.resume
    
    configs_list = [OmegaConf.load(cfg) for cfg in opt.ldm]
    cli = OmegaConf.from_dotlist(unknown)
    configs = OmegaConf.merge(*configs_list, cli)

    gpu = True
    eval_mode = True

    ldm_model, global_step = load_model(configs, ckpt, gpu, eval_mode)
    print(f"VQ autoencoder loaded from {configs_list[0].model.params.first_stage_config.params.ckpt_path}")
    #-------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    if opt.xcat:
        from utils.utils import Dataset_XCAT_Registration
        train_loader = Data.DataLoader(
            Dataset_XCAT_Registration(
                data_root=opt.xcat_path, split='train',
                motion_types=['identity', 'rotate10', 'scale05', 'warp'],  # 添加 scale05
                flip_p=0.5,
            ),
            batch_size=bs, shuffle=True, num_workers=0
        )
        val_loader = Data.DataLoader(
            Dataset_XCAT_Registration(
                data_root=opt.xcat_path, split='val',
                motion_types=['identity'],
                flip_p=0.0,
            ),
            batch_size=bs, shuffle=False, num_workers=0
        )
        test_loader = Data.DataLoader(
            Dataset_XCAT_Registration(
                data_root=opt.xcat_path, split='test',
                motion_types=['identity'],
                flip_p=0.0,
            ),
            batch_size=bs, shuffle=False, num_workers=0
        )
        print(f"XCAT mode: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    elif opt.fixed_motion:
        from ldm.data.xcat_Motion_Seq import XCATSeqRegistration
        train_loader = Data.DataLoader(
            XCATSeqRegistration(data_root=opt.fixed_motion_path, split='train', flip_p=0.5),
            batch_size=bs, shuffle=True, num_workers=0
        )
        val_loader = Data.DataLoader(
            XCATSeqRegistration(data_root=opt.fixed_motion_path, split='val', flip_p=0.0),
            batch_size=bs, shuffle=False, num_workers=0
        )
        test_loader = Data.DataLoader(
            XCATSeqRegistration(data_root=opt.fixed_motion_path, split='test', flip_p=0.0),
            batch_size=bs, shuffle=False, num_workers=0
        )
        print(f"FixedMotion mode: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    else:
        train_loader = Data.DataLoader(
            XCATNPZRegistration(data_root=datapath, split='train', flip_p=0.5),
            batch_size=bs, shuffle=True, num_workers=1
        )
        val_loader = Data.DataLoader(
            XCATNPZRegistration(data_root=datapath, split='val', flip_p=0.0),
            batch_size=bs, shuffle=False, num_workers=1
        )
        test_loader = Data.DataLoader(
            XCATNPZRegistration(data_root=datapath, split='test', flip_p=0.0),
            batch_size=bs, shuffle=False, num_workers=1
        )
        print(f"NPZ mode: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    model = LDMMorph.LDMMorph(128*2,192*2,320*2,448*2, use_ldm=not opt.no_ldm)
    model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    loss_similarity_ncc = ncc_loss
    loss_similarity_mse = MSE().loss
    loss_smooth = smoothloss

    transform = SpatialTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if opt.xcat:
        model_dir = f'./logs/XCAT_TransMorph_Smooth_{smooth}_beta_{beta}_xcat_motion_604/'
        csv_name  = f'./logs/XCAT_TransMorph_Smooth_{smooth}_beta_{beta}_xcat_motion_604.csv'
    elif opt.fixed_motion:
        model_dir = f'./logs/FixedMotion_TransMorph_Smooth_{smooth}_beta_{beta}_624/'
        csv_name  = f'./logs/FixedMotion_TransMorph_Smooth_{smooth}_beta_{beta}_624.csv'
    else:
        model_dir = './logs/TransScorelm_Smooth_0.4_beta_0.8_7_15_15_624/'
        csv_name  = './logs/TransScorelm_Smooth_0.4_beta_0.8_7_15_15_624.csv'

    # CSV表头：根据 loss_type 显示相应列
    f = open(csv_name, 'w')
    with f:
        if opt.loss_type == 'ncc':
            fnames = ['Index', 'NCC_Val_S', 'OrgNCC_Val_S', 'NCC_Test', 'OrgNCC_Test']
        else:
            fnames = ['Index', 'MSE_Val_S', 'NCC_Val_S', 'MSE_Test', 'NCC_Test']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    lossall = np.zeros((3, iteration+1))
    step = 1
    epoch = 0
    csv_dice = 0
    while step <= iteration:
        for X, Y, segx, segy, _ in train_loader:

            X = X.cuda().float()
            Y = Y.cuda().float()
            
            # 调试信息：输入数据范围
            if step == 1 or step % 50 == 0:
                print(f'\n[Step {step}] X range: [{X.min():.4f}, {X.max():.4f}], Y range: [{Y.min():.4f}, {Y.max():.4f}]')

            # =========================================================
            # [Ablation] LDM 特征生成：--no_ldm 时仍然计算（供 latent loss 使用），
            # 但模型内部会忽略它们（用 CNN 特征替代）
            # =========================================================
            mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()
            fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()

            noise = None
            noise = default(noise, lambda: torch.randn_like(mov_z))
            x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
            y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([t_enc]).cuda(), noise=noise)

            outx = ldm_model.apply_model(x_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
            outy = ldm_model.apply_model(y_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)

            score0 = torch.cat((outx[1][0][0],  outx[1][0][2], outy[1][0][0],  outy[1][0][2]),  dim=1)
            score1 = torch.cat((outx[1][0][3],  outx[1][0][5], outy[1][0][3],  outy[1][0][5]),  dim=1)
            score2 = torch.cat((outx[1][0][6],  outx[1][0][8], outy[1][0][6],  outy[1][0][8]),  dim=1)
            score3 = torch.cat((outx[1][0][9],  outx[1][0][11], outy[1][0][9],  outy[1][0][11]),  dim=1)
            if step == 1:
                print('score0:', score0.shape)
                print('score1:', score1.shape)
                print('score2:', score2.shape)
                print('score3:', score3.shape)
            D_f_xy = model(X, Y, score0, score1, score2, score3)
            _, X_Y = transform(X, D_f_xy.permute(0, 2, 3, 1))
            
            # 调试信息：形变场范围
            if step == 1 or step % 50 == 0:
                print(f'[Step {step}] D_f_xy range: [{D_f_xy.min():.6f}, {D_f_xy.max():.6f}], D_f_xy mean: {D_f_xy.mean():.6f}, D_f_xy std: {D_f_xy.std():.6f}')

            # [Ablation] 潜空间 MSE loss：--no_ldm 时仍然计算（LDM encoder 始终存在）
            mov_z_warped = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X_Y)).detach()
            loss_mse_latent = loss_similarity_mse(mov_z_warped, fix_z)

            if opt.loss_type == 'ncc':
                loss_image = ncc_loss(X_Y, Y)
            else:
                loss_image = F.mse_loss(X_Y, Y)

            loss1 = beta * loss_image + (1 - beta) * loss_mse_latent
            # print('beta:', beta)
            loss2 = loss_smooth(D_f_xy)
            # print('smooth:', smooth)
            loss = loss1 + smooth * loss2

            # 可视化：每 1000 步生成一张图
            if step % 1000 == 0:
                with torch.no_grad():
                    import matplotlib.pyplot as plt
                    X_cpu = X[0, 0].cpu().numpy()
                    Y_cpu = Y[0, 0].cpu().numpy()
                    XY_cpu = X_Y[0, 0].cpu().numpy()
                    D_cpu = D_f_xy[0].cpu().numpy()

                    # 计算雅可比行列式
                    D_disp = D_f_xy[0].cpu().numpy()  # [2, H, W]
                    dy = np.gradient(D_disp[0], axis=0)
                    dx = np.gradient(D_disp[0], axis=1)
                    dyy = np.gradient(D_disp[1], axis=0)
                    dxx = np.gradient(D_disp[1], axis=1)
                    jac_det = (1 + dx) * (1 + dyy) - dy * dxx

                    n_foldings = np.sum(jac_det < 0)
                    min_jac = jac_det.min()

                    diff_before = np.abs(X_cpu - Y_cpu)
                    diff_after = np.abs(XY_cpu - Y_cpu)

                    ncc_standard = 1.0 - ncc_loss(X_Y, Y).item()
                    loss_image_val = loss_image.item()
                    mse_standard = F.mse_loss(X_Y, Y).item()

                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    titles = ['Moving (X)', 'Fixed (Y)', 'Warped (X→Y)', 'Diff Before',
                              'Diff After', 'Jac Det (fold={}, min={:.3f})'.format(n_foldings, min_jac), 'Disp Field-X', 'Disp Field-Y']
                    imgs = [X_cpu, Y_cpu, XY_cpu, diff_before,
                            diff_after, jac_det, D_cpu[0], D_cpu[1]]
                    for ax, img, title in zip(axes.flat, imgs, titles):
                        if 'Jac' in title or 'Disp' in title:
                            ax.imshow(img, cmap='RdBu', vmin=-0.5, vmax=1.5)
                        else:
                            img_vmax = max(X_cpu.max(), Y_cpu.max(), XY_cpu.max())
                            ax.imshow(img, cmap='gray', vmin=0, vmax=img_vmax)
                        ax.set_title(title, fontsize=10)
                        ax.axis('off')

                    if opt.loss_type == 'ncc':
                        suptitle_str = f'[Step {step}] loss={loss.item():.4f}  NCC_train={loss_image_val:.4f}  MSE_z={loss_mse_latent.item():.4f}'
                    else:
                        suptitle_str = f'[Step {step}] loss={loss.item():.4f}  MSE_train={loss_image_val:.4f}  MSE_z={loss_mse_latent.item():.4f}  NCC_ref={ncc_standard:.4f}'
                    fig.suptitle(suptitle_str, fontsize=12)
                    plt.tight_layout()
                    fig.savefig(f'{model_dir}vis_step_{step:06d}.png', dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    print(f'\n[Visualization] Saved to {model_dir}vis_step_{step:06d}.png')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            lossall[:,step] = np.array([loss.item(), loss1.item(), loss2.item()])
            loss_name = f'{opt.loss_type.upper()}'
            weighted_latent_loss = (1-beta) * loss_mse_latent.item()
            weighted_ncc_loss = beta * loss_image.item()
            sys.stdout.write("\r" + 'step "{0}" -> train loss "{1:.4f}" - {3} "{2:.4f}" - weighted_MSE_z "{4:.4f}" - weighted_NCC_z "{5:.4f}" - smh "{6:.4f}"'.format(step, loss.item(), loss_image.item(), loss_name, weighted_latent_loss, weighted_ncc_loss, loss2.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                with torch.no_grad():
                    # 验证集：根据 loss_type 计算相应的 loss
                    Val_Loss_List = []
                    NCCs_Val_NCC = []
                    NCCs_Val_Org = []
                    
                    for xv, yv, xv_seg, yv_seg, _ in val_loader:

                        xv, yv, xv_seg, yv_seg = xv.to(device), yv.to(device), xv_seg.to(device), yv_seg.to(device)
                        
                        model.eval()

                        # =========================================================
                        # [Ablation] 验证集：LDM 特征始终计算（latent loss 仍生效）
                        # =========================================================
                        vmov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(xv)).detach()
                        vfix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(yv)).detach()

                        noise_v = torch.randn_like(vmov_z)
                        vx_noisy = ldm_model.q_sample(x_start=vmov_z, t=torch.tensor([t_enc]).cuda(), noise=noise_v)
                        vy_noisy = ldm_model.q_sample(x_start=vfix_z, t=torch.tensor([t_enc]).cuda(), noise=noise_v)

                        voutx = ldm_model.apply_model(vx_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
                        vouty = ldm_model.apply_model(vy_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)

                        vscore0 = torch.cat((voutx[1][0][0],  voutx[1][0][2], vouty[1][0][0],  vouty[1][0][2]),  dim=1)
                        vscore1 = torch.cat((voutx[1][0][3],  voutx[1][0][5], vouty[1][0][3],  vouty[1][0][5]),  dim=1)
                        vscore2 = torch.cat((voutx[1][0][6],  voutx[1][0][8], vouty[1][0][6],  vouty[1][0][8]),  dim=1)
                        vscore3 = torch.cat((voutx[1][0][9],  voutx[1][0][11], vouty[1][0][9],  vouty[1][0][11]),  dim=1)

                        Dv_f_xy = model(xv, yv, vscore0, vscore1, vscore2, vscore3)
                        _, warped_xv = transform(xv, Dv_f_xy.permute(0, 2, 3, 1))

                        for bs_index in range(bs):
                            # 根据 loss_type 计算相应的图像域 loss
                            if opt.loss_type == 'ncc':
                                loss_val = 1.0 - ncc_loss(
                                    yv[bs_index,...].unsqueeze(0),
                                    warped_xv[bs_index,...].unsqueeze(0).detach()
                                ).item()
                            else:
                                loss_val = F.mse_loss(
                                    yv[bs_index,...].unsqueeze(0),
                                    warped_xv[bs_index,...].unsqueeze(0).detach()
                                ).item()

                            # 标准NCC（配准后，用于衡量配准质量）
                            ncc_s = 1.0 - ncc_loss(
                                yv[bs_index,...].unsqueeze(0),
                                warped_xv[bs_index,...].unsqueeze(0).detach()
                            ).item()

                            # 标准NCC（配准前）
                            ncc_org_s = 1.0 - ncc_loss(
                                yv[bs_index,...].unsqueeze(0),
                                xv[bs_index,...].unsqueeze(0).detach()
                            ).item()

                            Val_Loss_List.append(loss_val)
                            NCCs_Val_NCC.append(ncc_s)
                            NCCs_Val_Org.append(ncc_org_s)

                    # 计算平均值
                    csv_loss_s = np.mean(Val_Loss_List)
                    csv_ncc_s = np.mean(NCCs_Val_NCC)
                    csv_ncc_org_s = np.mean(NCCs_Val_Org)

                    modelname = 'NCCVal_{:.4f}_Epoch_{:04d}.pth'.format(csv_ncc_s, step)
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                    np.save(model_dir + 'Loss.npy', lossall)

                    print(f'\n    [Validation] {opt.loss_type.upper()}_S: {csv_loss_s:.4f}  '
                          f'NCC_S: {csv_ncc_s:.4f}  OrgNCC_S: {csv_ncc_org_s:.4f}')
                    print(f'    Delta_S: {csv_ncc_s - csv_ncc_org_s:+.4f}')

                    # CSV 记录
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        if opt.loss_type == 'ncc':
                            writer.writerow([step, csv_loss_s, csv_ncc_org_s, -1, -1])
                        else:
                            writer.writerow([step, csv_loss_s, csv_ncc_s, -1, -1])

                    model.train()

            step += 1

            if step > iteration:
                break
        print("one epoch pass")

    np.save(model_dir + '/Loss.npy', lossall)

    # ==================== 训练结束后评估测试集 ====================
    if test_loader is not None:
        print("\n" + "="*60)
        print("Final Test Set Evaluation (after training, no leakage)")
        print("="*60)
        model.eval()
        Test_Loss_List = []
        NCCs_Test = []
        NCCs_Test_org = []
        with torch.no_grad():
            for xt, yt, _, _, _ in test_loader:
                xt, yt = xt.to(device), yt.to(device)

                # =========================================================
                # [Ablation] 测试集：LDM 特征始终计算（latent loss 仍生效）
                # =========================================================
                tmov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(xt)).detach()
                tfix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(yt)).detach()
                tx_noisy = ldm_model.q_sample(x_start=tmov_z, t=torch.tensor([t_enc]).cuda(), noise=torch.randn_like(tmov_z))
                ty_noisy = ldm_model.q_sample(x_start=tfix_z, t=torch.tensor([t_enc]).cuda(), noise=torch.randn_like(tfix_z))
                toutx = ldm_model.apply_model(tx_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
                touty = ldm_model.apply_model(ty_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
                tscore0 = torch.cat((toutx[1][0][0], toutx[1][0][2], touty[1][0][0], touty[1][0][2]), dim=1)
                tscore1 = torch.cat((toutx[1][0][3], toutx[1][0][5], touty[1][0][3], touty[1][0][5]), dim=1)
                tscore2 = torch.cat((toutx[1][0][6], toutx[1][0][8], touty[1][0][6], touty[1][0][8]), dim=1)
                tscore3 = torch.cat((toutx[1][0][9], toutx[1][0][11], touty[1][0][9], touty[1][0][11]), dim=1)

                Dt_f_xy = model(xt, yt, tscore0, tscore1, tscore2, tscore3)
                _, warped_xt = transform(xt, Dt_f_xy.permute(0, 2, 3, 1))
                for bs_index in range(bs):
                    # 根据 loss_type 计算相应的图像域 loss
                    if opt.loss_type == 'ncc':
                        loss_t = 1.0 - ncc_loss(yt[bs_index, ...].unsqueeze(0), warped_xt[bs_index, ...].unsqueeze(0).detach()).item()
                    else:
                        loss_t = F.mse_loss(yt[bs_index, ...].unsqueeze(0), warped_xt[bs_index, ...].unsqueeze(0).detach()).item()
                    
                    ncc_t = 1.0 - ncc_loss(yt[bs_index, ...].unsqueeze(0), warped_xt[bs_index, ...].unsqueeze(0).detach()).item()
                    ncc_t_org = 1.0 - ncc_loss(yt[bs_index, ...].unsqueeze(0), xt[bs_index, ...].unsqueeze(0).detach()).item()
                    Test_Loss_List.append(loss_t)
                    NCCs_Test.append(ncc_t)
                    NCCs_Test_org.append(ncc_t_org)
        print(f"\n    [Test] {opt.loss_type.upper()}: {np.mean(Test_Loss_List):.4f}  NCC: {np.mean(NCCs_Test):.4f}  OrgNCC: {np.mean(NCCs_Test_org):.4f}  Delta: {np.mean(NCCs_Test) - np.mean(NCCs_Test_org):+.4f}")
        print(f"    Test samples: {len(NCCs_Test)}")

        # 将测试结果追加到 CSV 最后一行
        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            if opt.loss_type == 'ncc':
                writer.writerow(['FINAL_TEST', -1, -1, np.mean(NCCs_Test), np.mean(NCCs_Test_org)])
            else:
                writer.writerow(['FINAL_TEST', np.mean(Test_Loss_List), -1, np.mean(NCCs_Test), np.mean(NCCs_Test_org)])
        print(f"\nTest results appended to: {csv_name}")

train()