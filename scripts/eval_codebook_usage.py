"""
计算 XCAT VQ16 码本利用率：
  - 加载训练好的 VQModel
  - 在训练集 718 对上跑 encode()，统计 16384 个 code 的使用频率
  - 报告：usage ratio (used codes / total)，top-k 集中度，Gini 系数

输出到 stdout + 一份 JSON 到 logs/.../codebook_usage.json
"""
import os, sys, json, glob, time
sys.path.insert(0, '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main')

import numpy as np
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

CKPT = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-04-18T12-44-46_autoencoder_xcat_vq16_70_15_15/checkpoints/last.ckpt'
CFG  = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/configs/autoencoder/xcat-autoencoder-vq16.yaml'
XCAT_FIXED = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data/fixed/fixed'
XCAT_MOVING = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/xcat_data/moving/moving'
OUT_JSON = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-04-18T12-44-46_autoencoder_xcat_vq16_70_15_15/codebook_usage.json'
LOG_PATH = '/root/autodl-tmp/LDM-Morph-main/LDM-Morph-main/logs/2026-04-18T12-44-46_autoencoder_xcat_vq16_70_15_15/codebook_usage.log'


def natsorted(lst):
    import re
    return sorted(lst, key=lambda s: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)])


def gini(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[arr > 0].sort() if False else np.sort(arr)
    arr = arr[arr > 0]
    n = len(arr)
    if n == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * arr) - (n + 1) * arr.sum()) / (n * arr.sum())


def main():
    cfg = OmegaConf.load(CFG)
    model = instantiate_from_config(cfg.model)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    miss, unx = model.load_state_dict(ckpt['state_dict'], strict=False)
    print(f'[load] missing={len(miss)}, unexpected={len(unx)}, step={ckpt.get("global_step")}, epoch={ckpt.get("epoch")}')
    # 默认 GPU；GPU 显存不够或 EVAL_CPU=1 时回退 CPU
    force_cpu = os.environ.get('EVAL_CPU', '0') == '1'
    if force_cpu or not torch.cuda.is_available():
        device = 'cpu'
    else:
        try:
            torch.cuda.empty_cache()
            free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3
            if free > 4.0:
                device = 'cuda'
            else:
                print(f'[warn] GPU only {free:.1f}GB free, falling back to CPU')
                device = 'cpu'
        except Exception:
            device = 'cpu'
    model = model.to(device).eval()
    # 关闭 attention 中 checkpointing 让推理快一些（用标准 attention）
    for m in model.modules():
        if hasattr(m, 'attn_1'):
            m.attn_1.checkpoint = False
        if hasattr(m, 'attn_2'):
            m.attn_2.checkpoint = False
    n_embed = cfg.model.params.n_embed
    print(f'[info] n_embed = {n_embed}, device = {device}')

    fx_paths = natsorted(glob.glob(os.path.join(XCAT_FIXED, '*.npy')))
    mv_paths = natsorted(glob.glob(os.path.join(XCAT_MOVING, '*.npy')))
    n = min(len(fx_paths), len(mv_paths))
    # 训练按 70% 划分：718 对
    n_train = int(n * 0.7)
    fx_train = fx_paths[:n_train]
    mv_train = mv_paths[:n_train]
    # 调试子集
    n_subset = int(os.environ.get('EVAL_SUBSET', '0'))
    if n_subset > 0:
        print(f'[debug] EVAL_SUBSET={n_subset}, only evaluating first {n_subset} pairs')
        fx_train = fx_train[:n_subset]
        mv_train = mv_train[:n_subset]
        n_train = len(fx_train)
    print(f'[data] total pairs={n}, train pairs used={n_train}')

    usage = np.zeros(n_embed, dtype=np.int64)
    total_tokens = 0
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for i, (fp, mp) in enumerate(zip(fx_train, mv_train)):
            fix = np.load(fp).astype(np.float32)
            mov = np.load(mp).astype(np.float32)
            # 训练时 XCATTrain L318-322: 用 fixed 的 min/max 归一化 moving 也
            # L315 还对 moving 做 motion simulation，但这里我们用 raw moving 评估
            minv, maxv = fix.min(), fix.max()
            if maxv - minv > 1e-6:
                fix_t = (fix - minv) / (maxv - minv)
                mov_t = (mov - minv) / (maxv - minv)
            else:
                fix_t = fix
                mov_t = mov

            # 同时统计 fixed + moving（VQ 训练时两者都参与 codebook 更新）
            for img in (fix_t, mov_t):
                x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
                with torch.no_grad():
                    z = model.encoder(x)
                    h = model.quant_conv(z)
                    _, _, info = model.quantize(h)
                    idx = info[2]   # [B, h, w]
                idx_flat = idx.flatten().cpu().numpy()
                usage += np.bincount(idx_flat, minlength=n_embed)
                total_tokens += idx_flat.size
            n_done += 1
            if (i + 1) % 50 == 0:
                used = (usage > 0).sum()
                p = usage.astype(np.float64)
                p_sum = p.sum()
                if p_sum > 0:
                    p_nz = p[p > 0] / p_sum
                    ppl = float(np.exp(-(p_nz * np.log(p_nz)).sum()))
                else:
                    ppl = 0.0
                print(f'  [{n_done}/{n_train}]  used_codes={used}/{n_embed} ({100*used/n_embed:.2f}%)  '
                      f'ppl={ppl:.1f}/{n_embed}  top1_usage={usage.max()/max(1,usage.sum()):.4f}  '
                      f'elapsed={time.time()-t0:.1f}s')

    used = (usage > 0).sum()
    s = usage.sum()
    sorted_u = np.sort(usage)[::-1]
    topk = sorted_u[:5].tolist()
    # 计算最终 perplexity（标准写法：p*log(p) 再 exp）
    p = usage.astype(np.float64)
    p_sum = p.sum()
    if p_sum > 0:
        p_nz = p[p > 0] / p_sum
        ppl = float(np.exp(-(p_nz * np.log(p_nz)).sum()))
    else:
        ppl = 0.0
    report = {
        'ckpt': CKPT,
        'step': ckpt.get('global_step'),
        'epoch': ckpt.get('epoch'),
        'n_embed': int(n_embed),
        'n_pairs_evaluated': int(n_train),
        'images_per_pair': 2,    # fixed + moving 都统计
        'total_tokens': int(s),
        'used_codes': int(used),
        'usage_ratio': float(used / n_embed),
        'usage_ratio_pct': f'{100 * used / n_embed:.2f}%',
        'dead_codes': int(n_embed - used),
        'dead_codes_pct': f'{100 * (n_embed - used) / n_embed:.2f}%',
        'effective_codes_ppl': ppl,            # 码本使用的有效复杂度（越大越好，<= n_embed）
        'effective_codes_ppl_ratio': float(ppl / n_embed),
        'top1_usage_ratio': float(sorted_u[0] / max(1, s)),
        'top5_usage_ratio': float(sorted_u[:5].sum() / max(1, s)),
        'top10_usage_ratio': float(sorted_u[:10].sum() / max(1, s)),
        'top1_count': int(sorted_u[0]),
        'top5_counts': topk,
        'gini_coefficient': float(gini(usage)),
        'usage_distribution': {
            'used_once': int((usage == 1).sum()),
            'used_2_5': int(((usage >= 2) & (usage <= 5)).sum()),
            'used_6_20': int(((usage >= 6) & (usage <= 20)).sum()),
            'used_21_100': int(((usage >= 21) & (usage <= 100)).sum()),
            'used_101_1000': int(((usage >= 101) & (usage <= 1000)).sum()),
            'used_1000_plus': int((usage > 1000).sum()),
        }
    }
    print('\n========== CODEBOOK USAGE REPORT ==========')
    for k, v in report.items():
        if isinstance(v, dict):
            print(f'  {k}:')
            for kk, vv in v.items():
                print(f'    {kk}: {vv}')
        else:
            print(f'  {k}: {v}')
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n[done] saved report to {OUT_JSON}')


if __name__ == '__main__':
    main()