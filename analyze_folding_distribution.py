"""
分析折叠的空间分布：是在肝脏内还是背景？
"""
import os, sys, glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

sys.path.insert(0, '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main')
from ldm.data.sey_registration import SEYRegistration
from utils.utils import jacobian_determinant_vxm
import TransModels.LDMMorph as LDMMorph
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

# 配置
RESUME = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/SEY_Only_smooth1.0_beta0.8_bending0.0_jacdet2.0/NCCVal_0.6975_Epoch_15000.pth'
LDM_CKPT = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-07-04T14-32-23_sey-ldm-vq16-64ch/checkpoints/last.ckpt'
LDM_CFG = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/latent-diffusion/sey-ldm-vq16-64ch.yaml'
SEY_PATH = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/datasets/SEY/prep'
SAVE_DIR = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/folding_analysis/'
os.makedirs(SAVE_DIR, exist_ok=True)

def body_mask(img_tensor, thr=0.05):
    arr = img_tensor.detach().cpu().numpy()
    b, c, h, w = arr.shape
    out = np.zeros((b, c, h, w), dtype=bool)
    for bi in range(b):
        m = arr[bi, 0] > thr
        m = binary_fill_holes(m)
        lab, n = label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            m = (lab == int(sizes.argmax()))
        out[bi, 0] = m
    return torch.from_numpy(out).to(img_tensor.device)

# 加载模型
def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

configs_list = [OmegaConf.load(LDM_CFG)]
configs = OmegaConf.merge(*configs_list)
ldm_model = load_model_from_config(configs.model, {"state_dict": None})
pl_sd = torch.load(LDM_CKPT, map_location="cpu")
ldm_model = load_model_from_config(configs.model, pl_sd["state_dict"])
print(f"LDM loaded: {LDM_CKPT}")

model = LDMMorph.LDMMorph(128*2, 192*2, 320*2, 448*2, use_ldm=True).cuda()
state_dict = torch.load(RESUME, map_location="cuda")
model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"Reg model loaded: {RESUME}")

from utils.utils import SpatialTransform
transform = SpatialTransform().cuda()
for p in transform.parameters():
    p.requires_grad = False

# 数据
ds = SEYRegistration(data_root=SEY_PATH, split='test', normalize=False)

# 分析前10个
n_samples = 10
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle("折叠分布分析：肝脏区域 vs 背景区域\n(红色=肝脏内折叠, 蓝色=背景内折叠, 绿色=无折叠)", y=1.02)

for i in range(n_samples):
    X, Y, _, _, pairname = ds[i]
    X = X.unsqueeze(0).float().cuda()
    Y = Y.unsqueeze(0).float().cuda()

    with torch.no_grad():
        mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()
        fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()
        noise = torch.randn_like(mov_z)
        x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([1]).cuda(), noise=noise)
        y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([1]).cuda(), noise=noise)
        outx = ldm_model.apply_model(x_noisy, t=torch.tensor([1]).cuda(), cond=None, return_ids=True)
        outy = ldm_model.apply_model(y_noisy, t=torch.tensor([1]).cuda(), cond=None, return_ids=True)
        score0 = torch.cat((outx[1][0][0], outx[1][0][2], outy[1][0][0], outy[1][0][2]), dim=1)
        score1 = torch.cat((outx[1][0][3], outx[1][0][5], outy[1][0][3], outy[1][0][5]), dim=1)
        score2 = torch.cat((outx[1][0][6], outx[1][0][8], outy[1][0][6], outy[1][0][8]), dim=1)
        score3 = torch.cat((outx[1][0][9], outx[1][0][11], outy[1][0][9], outy[1][0][11]), dim=1)
        D_f_xy, _ = model(X, Y, score0, score1, score2, score3)

    dvf = D_f_xy.permute(0, 2, 3, 1).cpu().numpy()[0]
    jac_det = jacobian_determinant_vxm(dvf.transpose(2, 0, 1).astype(np.float32))

    # 前景 mask
    fg = body_mask(Y).squeeze().cpu().numpy().astype(bool)
    bg = ~fg

    # 折叠 mask
    foldings = jac_det < 0
    n_total = np.sum(foldings)
    n_in_liver = np.sum(foldings & fg)
    n_in_bg = np.sum(foldings & bg)

    print(f"[{i:02d}] {pairname}: 总折叠={n_total}, 肝脏内={n_in_liver} ({n_in_liver/n_total*100:.1f}%), "
          f"背景内={n_in_bg} ({n_in_bg/n_total*100:.1f}%)")

    # 可视化
    ax = axes.flat[i]
    liver_norm = np.zeros_like(jac_det)
    liver_norm[fg] = jac_det[fg]
    liver_norm[~fg] = np.nan

    bg_norm = np.zeros_like(jac_det)
    bg_norm[bg] = jac_det[bg]
    bg_norm[~bg] = np.nan

    # 用两个子图叠加
    im_bg = ax.imshow(np.where(bg, jac_det, np.nan), cmap='Blues', vmin=-1, vmax=2)
    im_liver = ax.imshow(np.where(fg, jac_det, np.nan), cmap='RdYlGn', vmin=-1, vmax=2)
    ax.set_title(f"{pairname}\n总折叠={n_total}, 肝内={n_in_liver}({n_in_liver/(n_total+1e-8)*100:.0f}%)", fontsize=9)
    ax.axis('off')

plt.tight_layout()
out_path = os.path.join(SAVE_DIR, 'folding_distribution.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out_path}")
