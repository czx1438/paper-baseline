"""
DPT-VQGAN: 双因子解耦 VQGAN
==============================================================
改动点（相对于 ldm/models/autoencoder.py 的 VQModel）：
  - 新增运动编码器（4层 stride-2 conv + 全局池化 → m_k）
  - 解剖编码器 E_a = 复用原 Encoder（VQ 量化后得到 a）
  - 解码器 G(a, m_k)：m_k 由 decoder 内部生成 FiLM (γ, β)
  - m_k 不量化，通过 FiLM 调制解码器各层

⚠️ 假设：原 Encoder 的 z_channels=1, z_shape=(1, 64, 64)
"""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import pytorch_lightning as pl
import torch.nn as nn
from contextlib import contextmanager

from ldm.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.model import Encoder, Normalize, nonlinearity
from ldm.util import instantiate_from_config


# ==================================================================
# 1. 运动编码器（加深版）
#    4 层 stride-2 conv：512→256→128→64→32，感受野覆盖全局运动
#    输出：干净的低维运动码 m_k ∈ R^{m_dim}
# ==================================================================
class MotionEncoder(nn.Module):
    """
    深层运动编码器。

    结构：4 层 stride-2 conv
        512×512 → conv(1→32, s=2) → 256×256
                   → conv(32→64, s=2) → 128×128
                   → conv(64→128, s=2) → 64×64
                   → conv(128→128, s=2) → 32×32
                   → AdaptiveAvgPool → Linear → m_k

    输出：m_k ∈ R^{m_dim}（干净的低维运动码，不含 gamma/beta）
    """

    def __init__(self, in_channels=1, out_channels=(32, 64, 128, 128), m_dim=64):
        super().__init__()
        ch = (in_channels,) + out_channels
        layers = []
        for i in range(len(ch) - 1):
            layers.append(nn.Conv2d(ch[i], ch[i+1], kernel_size=3, stride=2, padding=1))
            layers.append(nn.SiLU())
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(out_channels[-1], m_dim)

    def forward(self, x):
        """
        x: (B, 1, 512, 512)
        Returns: m_k (B, m_dim)
        """
        h = self.backbone(x)                # (B, 128, 32, 32)
        h = self.pool(h)                    # (B, 128, 1, 1)
        h = h.reshape(h.shape[0], -1)       # (B, 128) — 显式 reshape
        return self.proj(h)                  # (B, m_dim)


# ==================================================================
# 2. FiLM Decoder
#    输入：a (B, 1, 64, 64) + m_k (B, m_dim)
#    内部生成 FiLM γ/β，调制各层特征
# ==================================================================
class FiLMResBlock(nn.Module):
    """
    FiLM 调制的 ResBlock。
    输入通道数 = block_in（含 concat skip 后的总通道）。
    γ/β 在外部生成后传入（来自同一 resolution level）。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(0.0)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                Normalize(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, gamma, beta):
        """
        x: (B, in_channels, H, W)
        gamma, beta: (B, out_channels)
        """
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        # FiLM 调制
        g = gamma.unsqueeze(-1).unsqueeze(-1)
        b = beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + g) + b

        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)

        return h + self.shortcut(x)


class FiLMDecoder(nn.Module):
    """
    简化的 FiLM 解码器（无 skip connection）。

    结构：z (64x64) → Conv + ResBlocks × 3 levels → 512x512
    m_k 由 decoder 内部各层自己的 Linear 生成 FiLM γ/β。

    优点：完全避免原 UNet decoder 的 skip channel 复杂度，
    3 层上采样（8x）足够重建 512x512 图像。
    """

    def __init__(self, *, ch=128, out_ch=1, ch_mult=(1, 2, 4), num_res_blocks=2,
                 resolution=512, z_channels=1, m_dim=64,
                 dropout=0.0, **ignorekwargs):
        super().__init__()
        self.num_levels = 3  # 固定 3 层（2^3=8 下采样）
        self.m_dim = m_dim

        # 通道配置
        ch_outs = [ch * 4, ch * 2, ch]  # [512, 256, 128]

        # FiLM 生成器：每层一个 m_dim → out_channels 的投影
        self.film_gamma = nn.ModuleList([nn.Linear(m_dim, c) for c in ch_outs])
        self.film_beta  = nn.ModuleList([nn.Linear(m_dim, c) for c in ch_outs])

        block_in = ch * ch_mult[-1]  # 512
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)

        self.levels = nn.ModuleList()
        for level, out_c in enumerate(ch_outs):
            level_module = nn.Module()
            level_module.resblocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_module.resblocks.append(FiLMResBlock(block_in, out_c))
                block_in = out_c
            if level < len(ch_outs) - 1:
                level_module.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode="nearest"),
                    nn.Conv2d(out_c, out_c, 3, padding=1)
                )
            self.levels.append(level_module)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, padding=1)

    def forward(self, z, m_k):
        """
        z: (B, z_channels, 64, 64) — 解剖特征 a
        m_k: (B, m_dim) — 运动码
        Returns: (B, 1, 512, 512)
        """
        h = self.conv_in(z)

        for level in range(len(self.levels)):
            # 生成该层的 FiLM 参数
            gamma = self.film_gamma[level](m_k)   # (B, out_channels)
            beta  = self.film_beta[level](m_k)

            for rb in self.levels[level].resblocks:
                h = rb(h, gamma, beta)

            if level < len(self.levels) - 1:
                h = self.levels[level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


# ==================================================================
# 3. DPT-VQGAN 主模型
# ==================================================================
class DPTVQGAN(pl.LightningModule):
    """
    双因子解耦 VQGAN。

    前向流程（训练时 group batch）：
        1. encode_anat(x_k) → z_a → quantize → a（解剖特征，VQ 量化）
        2. encode_motion(x_k) → m_k（运动码，连续，不量化）
        3. decode(a, m_k) → x_k_hat

    关键设计：
        - a 在同一被试的所有相位间共享（loss 驱动共享）
        - m_k 每相位独立，低维连续向量
        - m_k 通过 FiLM 调制解码器，不直接 concat 到输入

    兼容接口（供下游 LDM-Morph 使用）：
        - encode(x): 返回 quantize 后的 a（与原 VQModel 行为一致）
        - decode(a, m_k): 用 a + m_k 解码
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 m_dim=64,
                 num_phases=9,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False,
                 use_ema=False,
                 # 解耦损失权重
                 w_anat=1.0,
                 w_swap=1.0,
                 w_phase=0.5,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.m_dim = m_dim
        self.num_phases = num_phases

        # 解剖编码器（复用原 Encoder）
        self.encoder = Encoder(**ddconfig)

        # VQ 量化（只作用在 a 上）
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # 运动编码器（4层 stride-2）
        self.motion_encoder = MotionEncoder(m_dim=m_dim)

        # FiLM 解码器
        self.decoder = FiLMDecoder(
            ch=ddconfig["ch"],
            out_ch=ddconfig["out_ch"],
            z_channels=ddconfig["z_channels"],
            m_dim=m_dim,
        )

        # 损失模块（原版 VQLPIPSWithDiscriminator）
        self.loss = instantiate_from_config(lossconfig)

        # 相位预测 MLP（输入 = m_k，维度 m_dim）
        self.phase_mlp = nn.Sequential(
            nn.Linear(m_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_phases),
        )

        if colorize_nlabels is not None:
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range

        self.use_ema = use_ema
        if self.use_ema:
            from ldm.modules.ema import LitEma
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

        # 解耦损失权重
        self.w_anat = w_anat
        self.w_swap = w_swap
        self.w_phase = w_phase

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    # ---- 编码 / 解码接口 ----

    def encode_anat(self, x):
        """解剖编码 + VQ 量化 → a"""
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_motion(self, x):
        """运动编码 → m_k（连续向量）"""
        return self.motion_encoder(x)

    def decode(self, a, m_k, use_checkpoint=False):
        """
        FiLM 解码：a ∈ (B, C, H, W), m_k ∈ (B, m_dim)
        use_checkpoint: 为 True 时用 gradient checkpointing 省显存（训练用），反传时重算 activation
        """
        if use_checkpoint:
            a = torch.utils.checkpoint.checkpoint(
                self.post_quant_conv, a, use_reentrant=False)
            return torch.utils.checkpoint.checkpoint(
                self.decoder, a, m_k, use_reentrant=False)
        a = self.post_quant_conv(a)
        return self.decoder(a, m_k)

    def decode_a_only(self, a):
        """仅用 a 解码（兼容性接口，m_k 用零向量）"""
        B = a.shape[0]
        m_k = torch.zeros(B, self.m_dim, device=a.device, dtype=a.dtype)
        return self.decode(a, m_k)

    # ---- 核心前向（单图）----
    def forward_single(self, x):
        """单图前向，返回 (重建图, qloss, indices)"""
        a, qloss, info = self.encode_anat(x)
        m_k = self.encode_motion(x)
        xrec = self.decode(a, m_k)
        return xrec, qloss, info[2]

    def forward(self, input, return_pred_indices=False):
        """兼容原 VQModel 接口"""
        xrec, qloss, ind = self.forward_single(input)
        if return_pred_indices:
            return xrec, qloss, ind
        return xrec, qloss

    # ---- 兼容 LDM-Morph 的接口 ----
    def get_input(self, batch, k):
        """
        兼容原 VQModel.get_input。
        ⚠️ 假设：batch[k] shape = (B, 1, H, W)（单图模式）
        """
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    # ---- 解耦训练 Step ----
    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        解耦训练：每个 batch 是 (B, num_frames, 1, H, W) 的 group。

        ⚠️ 假设：batch["images"] shape = (B, num_frames, 1, H, W)
        ⚠️ 假设：batch["phases"] shape = (B, num_frames)，值为 0..num_phases-1
        """
        images = batch["images"]      # (B, num_frames, 1, H, W)
        phases = batch["phases"]       # (B, num_frames)

        B, num_frames = images.shape[:2]
        print(f"B: {B}, num_frames: {num_frames}")
        device = images.device

        # 重塑为 flat batch: (B*num_frames, 1, H, W)
        x_flat = images.reshape(-1, 1, images.shape[3], images.shape[4])
        phases_flat = phases.reshape(-1)

        # ---------- 1. 解剖编码（整 batch 一次） ----------
        a_flat, qloss_flat, _ = self.encode_anat(x_flat)
        qloss = qloss_flat.mean()
        C, H, W = a_flat.shape[1:]
        a_group = a_flat.reshape(B, num_frames, C, H, W)

        # ---------- 2. 运动编码（整 batch 一次） ----------
        m_flat = self.encode_motion(x_flat)  # (B*NF, m_dim)
        m_group = m_flat.reshape(B, num_frames, self.m_dim)

        # ---------- 3. 整 batch 一次解码 ----------
        xrec_flat = self.decode(a_flat, m_flat, use_checkpoint=True)
        H_out, W_out = xrec_flat.shape[2], xrec_flat.shape[3]
        xrec_group = xrec_flat.reshape(B, num_frames, 1, H_out, W_out)

        # ---------- 4. 解耦损失（5 个） ----------
        loss_dict = self._compute_decoupling_losses(
            x_group=images,
            xrec_group=xrec_group,
            a_group=a_group,
            m_group=m_group,
            phases_flat=phases_flat,
            qloss=qloss,
        )

        # ---------- 5. 按 optimizer_idx 分别返回 ----
        if optimizer_idx == 0:
            # AE optimizer：更新 encoder + decoder + VQ
            qloss_for_loss = qloss_flat.unsqueeze(0)
            loss_ae, log_dict = self.loss(
                qloss_for_loss,
                x_flat,
                xrec_flat,
                optimizer_idx=0,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            loss_ae = (
                loss_ae +
                self.w_anat * loss_dict["L_anat"] +
                self.w_swap * loss_dict["L_swap"] +
                self.w_phase * loss_dict["L_phase"] +
                1.0 * loss_dict["L_phase_contrastive"].detach()
            )
            log_dict["train/total_loss"] = loss_ae.detach()
            log_dict["train/L_anat"] = loss_dict["L_anat"].detach()
            log_dict["train/L_swap"] = loss_dict["L_swap"].detach()
            log_dict["train/L_phase"] = loss_dict["L_phase"].detach()
            log_dict["train/L_phase_contrastive"] = loss_dict["L_phase_contrastive"].detach()
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return loss_ae

        if optimizer_idx == 1:
            # Discriminator
            loss_disc, log_dict = self.loss(
                None,
                x_flat,
                xrec_flat.detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                split="train",
            )
            self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return loss_disc

        if optimizer_idx == 2:
            # Motion optimizer：L_phase + L_phase_contrastive，隔离 L_rec 梯度淹没
            L_motion = loss_dict["L_phase"] + 1.0 * loss_dict["L_phase_contrastive"]
            self.log("train/L_phase_motion_only",
                     (loss_dict["L_phase"] + 1.0 * loss_dict["L_phase_contrastive"]).detach(),
                     prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return L_motion

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """
        AE optimizer（idx=0）不更新 motion 参数，防止 L_rec 梯度淹没 L_phase。
        motion 参数只由 optimizer idx=2 更新。
        """
        opt_ae, opt_motion, opt_disc = self.optimizers()
        if optimizer is opt_ae:
            motion_params = set(self.motion_encoder.parameters()) | set(self.phase_mlp.parameters())
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p in motion_params and p.grad is not None:
                        p.grad = None

    def _compute_decoupling_losses(self, x_group, xrec_group, a_group,
                                   m_group, phases_flat, qloss):
        """
        计算解耦损失（5 个）。
        """
        B, num_frames = x_group.shape[:2]

        # ---- L_rec：重建误差 ----
        L_rec = F.l1_loss(xrec_group, x_group)

        # ---- L_anat：同被试不同相位的 a 应一致 ----
        # 用组内均值作为锚，比固定第0帧更对称稳定
        a_mean = a_group.mean(dim=1, keepdim=True)   # (B, 1, C, H, W)
        a_expanded = a_mean.expand(B, num_frames, -1, -1, -1)  # (B, NF, C, H, W)
        L_anat = F.l1_loss(a_expanded, a_group)

        # ---- L_swap：核心解耦损失 ----
        # 随机抽 2 个相位做交叉重建（显存降 4×，长期覆盖所有相位）
        # a_fixed.detach() 切断解剖编码器梯度，m_other 保持连接 → 梯度流到 motion_encoder 和 decoder
        # decode 用 gradient checkpointing：前向不存 activation，省显存且梯度不变
        a_fixed = a_group[:, 0].detach()                     # (B, C, Ha, Wa)
        Ha, Wa = a_fixed.shape[2], a_fixed.shape[3]
        # 随机选 2 个相位（避免每次选相同相位导致过拟合）
        k_swap = 2
        idxs = torch.randperm(num_frames - 1, device=a_fixed.device)[:k_swap]   # (k_swap,)
        a_fixed_k = (a_fixed.unsqueeze(1)                                         # (B, 1, C, Ha, Wa)
                     .expand(B, k_swap, -1, -1, -1)                              # (B, k, C, Ha, Wa)
                     .reshape(B * k_swap, -1, Ha, Wa))                           # (B*k, C, Ha, Wa)
        m_k = m_group[:, 1:idxs.max()+2][:, idxs].reshape(B * k_swap, -1)       # (B*k, m_dim)
        x_swap_flat = self.decode(a_fixed_k, m_k, use_checkpoint=True)             # (B*k, 1, H, W)
        x_target_flat = x_group[:, 1:][:, idxs].reshape(B * k_swap, 1,           # (B*k, 1, H, W)
                                                        x_group.shape[3], x_group.shape[4])
        L_swap = F.l1_loss(x_swap_flat, x_target_flat)

        # ---- L_phase：m_k 应携带相位信息 ----
        # m_flat: (B*NF, m_dim)
        m_flat = m_group.reshape(-1, self.m_dim)
        L_phase = F.cross_entropy(self.phase_mlp(m_flat), phases_flat)

        # ---- L_phase_contrastive：相位对比损失（NT-Xent）----
        # 同一相位（不同被试）的 m_k 应接近，不同相位的应远离
        # phases_flat: (B*NF,)  int64
        # m_flat:       (B*NF, m_dim)
        N = m_flat.shape[0]
        P = F.normalize(m_flat, dim=1)
        S = torch.mm(P, P.T)
        labels = phases_flat.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        mask_pos = mask_pos.clone()
        mask_pos.fill_diagonal_(0.0)

        # batch 内必须有至少一组同相位样本（batch_size >= 2 才有意义）
        # 否则 exp_pos 全 0，log(eps/eps) = 常数，无 grad_fn
        has_pos = mask_pos.sum() > 0
        if has_pos:
            eps = 1e-8
            exp_pos = (torch.exp(S / 0.1) * mask_pos).sum(1, keepdim=True).clamp(min=eps)
            exp_neg = (torch.exp(S / 0.1) * (labels != labels.T).float()).sum(1, keepdim=True).clamp(min=eps)
            L_phase_contrastive = (-torch.log(exp_pos / (exp_pos + exp_neg))).mean()
        else:
            # 无正样本对时返回零 loss（用 m_flat 保持 grad_fn）
            L_phase_contrastive = m_flat.sum() * 0

        return {
            "L_rec": L_rec,
            "L_anat": L_anat,
            "L_swap": L_swap,
            "L_phase": L_phase,
            "L_phase_contrastive": L_phase_contrastive,
        }

    def validation_step(self, batch, batch_idx):
        images = batch["images"]         # (B, NF, 1, H, W)
        phases = batch["phases"]         # (B, NF)
        B, num_frames = images.shape[:2]
        H, W = images.shape[3], images.shape[4]

        # VQ quantize 逐帧处理（避免 9 张图同时算 16384 codebook 距离矩阵 OOM）
        xrec_list = []
        qloss_list = []
        for f in range(num_frames):
            x_f = images[:, f]                        # (B, 1, H, W)
            xrec_f, qloss_f, _ = self.forward_single(x_f)   # 逐帧过 quantize
            xrec_list.append(xrec_f)
            qloss_list.append(qloss_f.mean())
        xrec_flat = torch.cat(xrec_list, dim=0)       # (B*NF, 1, H, W)
        qloss = torch.stack(qloss_list).mean()

        x_flat = images.reshape(-1, 1, H, W)
        rec_loss = F.l1_loss(xrec_flat, x_flat)
        self.log("val/rec_loss", rec_loss, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)

        # 相位可分性（motion 编码和 MLP 可以 batch）
        m_flat = self.encode_motion(x_flat)
        phases_flat = phases.reshape(-1)
        phase_acc = (self.phase_mlp(m_flat).argmax(1) == phases_flat).float().mean()
        self.log("val/phase_acc", phase_acc, prog_bar=True, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)

        # 运动互换可视化（每 batch_idx=0 时做一次）
        if batch_idx == 0 and self.trainer is not None:
            self._log_motion_swap(batch, xrec_flat)

        return {"val/rec_loss": rec_loss}

    def _log_motion_swap(self, batch, xrec_flat):
        """
        固定第0帧的 a，逐帧换上其它相位的 m_j，decode 并保存可视化。
        逐帧过 VQ quantize（避免 9 张图同时算 16384 codebook 距离矩阵 OOM）。
        用于人工核对解耦是否真的发生。
        """
        images = batch["images"].to(self.device)   # (B, NF, 1, H, W)
        B, num_frames = images.shape[:2]
        H, W = images.shape[3], images.shape[4]

        with torch.no_grad():
            # 逐帧过 encode_anat（避免 quantize 距离矩阵 OOM）
            a_list = []
            for f in range(num_frames):
                a_f, _, _ = self.encode_anat(images[:, f])   # (B, C, Ha, Wa)
                a_list.append(a_f)
            a_flat = torch.cat(a_list, dim=0)                 # (B*NF, C, Ha, Wa)

            # motion 可以 batch（轻量）
            x_flat = images.reshape(-1, 1, H, W)
            m_flat = self.encode_motion(x_flat)                # (B*NF, m_dim)

            Ha, Wa = a_flat.shape[2], a_flat.shape[3]
            a_group = a_flat.reshape(B, num_frames, -1, Ha, Wa)
            m_group = m_flat.reshape(B, num_frames, -1)

            # 取第一个被试
            a_fixed = a_group[0, 0:1]           # (1, C, Ha, Wa)
            ref_img  = images[0, 0:1]            # (1, 1, H, W) — I_0

            # G(a_fixed, m_j) for all j > 0（decoder 一次 batch）
            m_other = m_group[0, 1:]             # (NF-1, m_dim)
            a_fixed_exp = a_fixed.expand(num_frames-1, -1, -1, -1)  # (NF-1, C, Ha, Wa)
            xswap_flat = self.decode(a_fixed_exp, m_other)  # (NF-1, 1, H, W)
            xswap = xswap_flat.reshape(num_frames-1, 1, H, W)

            # 原始相位图
            orig_imgs = images[0, 1:]            # (NF-1, 1, H, W)

        # 保存可视化
        save_dir = self.trainer.default_root_dir or "./logs"
        step = self.global_step
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        swap_dir = os.path.join(save_dir, "motion_swap_viz")
        os.makedirs(swap_dir, exist_ok=True)

        n = num_frames - 1
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        for j in range(n):
            vmax = orig_imgs[j].max().item()
            axes[j, 0].imshow(ref_img[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
            axes[j, 0].set_title(f"Ref: I_0")
            axes[j, 0].axis('off')

            axes[j, 1].imshow(xswap[j, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
            mse = ((xswap[j] - orig_imgs[j])**2).mean().item()
            psnr = 0 if mse == 0 else 20 * np.log10(1.0 / max(mse**0.5, 1e-6))
            axes[j, 1].set_title(f"G(a_0, m_{j+1}) PSNR={psnr:.1f}dB")
            axes[j, 1].axis('off')

            axes[j, 2].imshow(orig_imgs[j, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=vmax)
            axes[j, 2].set_title(f"GT: I_{j+1}")
            axes[j, 2].axis('off')

        plt.suptitle(f"Step {step} | Motion Swap | Subject 0", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{swap_dir}/step{step:06d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[MotionSwap] Saved to {swap_dir}/step{step:06d}.png")

    def configure_optimizers(self):
        lr = self.learning_rate
        lr_g = self.lr_g_factor * lr
        lr_m = 5.0 * lr_g  # motion 相关参数用更高学习率，对抗梯度淹没

        # AE optimizer（encoder + decoder + VQ）
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quantize.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr_g, betas=(0.5, 0.9)
        )

        # Motion optimizer（单独隔离梯度，避免被 L_rec 淹没）
        opt_motion = torch.optim.Adam(
            list(self.motion_encoder.parameters()) +
            list(self.phase_mlp.parameters()),
            lr=lr_m, betas=(0.5, 0.9)
        )

        # Discriminator optimizer
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )

        return [opt_ae, opt_motion, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        images = batch["images"]  # (B, num_frames, 1, H, W)
        x = images[0, :3].reshape(-1, 1, images.shape[3], images.shape[4])
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _, _ = self.forward_single(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema and self.use_ema:
            with self.ema_scope():
                xrec_ema, _, _ = self.forward_single(x)
                log["reconstructions_ema"] = xrec_ema
        return log
