import torch.nn as nn
import torch
import torch.nn.functional as F
import TransModels.Conv2dReLU as Conv2dReLU
import TransModels.LWSA as LWSA
import TransModels.LWCA as LWCA
import TransModels.Decoder as Decoder
import utils.configs as configs


class LDMMorph(nn.Module):
    def __init__(self, channel_1, channel_2, channel_3, channel_4, use_ldm=True,
                 use_motion_film=False):
        """
        Args:
            channel_1..channel_4 : Swin 4 个尺度通道数
            use_ldm              : 是否使用 LDM 特征 (CNN-only ablation 关闭)
            use_motion_film      : 是否使用 Image-conditioned Motion Embedding + FiLM
                                  - True  → 启用 motion encoder + FiLM
                                            (phase_id 参数被忽略)
                                  - False → 完全不用 FiLM, 行为等价 baseline
        """
        self.channel_1 = channel_1
        self.channel_2 = channel_2
        self.channel_3 = channel_3
        self.channel_4 = channel_4
        self.use_ldm = use_ldm
        self.use_motion_film = use_motion_film

        super(LDMMorph, self).__init__()

        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.ec1 = Conv2dReLU.Conv2dReLU(2, 32, 3, 1, use_batchnorm=False)
        self.start_channel = 64
        bias_opt = True

        config1 = configs.get_SelfAttention_config()
        config2 = configs.get_CrossAttention_config()

        self.lwsa = LWSA.LWSA(config1, in_channel=2)

        self.c1 = self.encoder(self.channel_1, self.start_channel,     bias=bias_opt)
        self.c2 = self.encoder(self.channel_2, self.start_channel * 2, bias=bias_opt)
        self.c3 = self.encoder(self.channel_3, self.start_channel * 4, bias=bias_opt)
        self.c4 = self.encoder(self.channel_4, self.start_channel * 8, bias=bias_opt)

        self.lwca1 = LWCA.LWCA(config2, dim_diy=64)
        self.lwca2 = LWCA.LWCA(config2, dim_diy=128)
        self.lwca3 = LWCA.LWCA(config2, dim_diy=256)
        self.lwca4 = LWCA.LWCA(config2, dim_diy=512)

        # =========================================================
        # [Ablation] CNN-only 编码器：当 use_ldm=False 时替代 LDM 特征进入 LWCA
        # 行为说明:
        #   - score0~4 仍然在 forward 中接收，latent loss 照常计算（依赖 score0）
        #   - 仅用 CNN 特征替换 LWCA 路径中的 c1~c4，验证 CNN 编码器能否替代 LDM
        # 输入: [B, 1, H, W]  (moving 或 fixed, H=W=512)
        # Swin LWSA 在 patch_embed 时做了 4x 下采样，所以 Swin 特征分辨率为:
        #   fea_4=[128,128], fea_8=[64,64], fea_16=[32,32], fea_32=[16,16]
        # CNN 编码器与 Swin 对齐 (初始下采样2x，每层再逐级下采样):
        #   m0: [B,32,H/2,W/2]  -> concat=64   -> proj_lwca0=64   -> lwca1 dim_diy=64
        #   m1: [B,64,H/4,W/4]  -> concat=128  -> proj_lwca1=128  -> lwca2 dim_diy=128
        #   m2: [B,128,H/8,W/8] -> concat=256  -> proj_lwca2=256  -> lwca3 dim_diy=256
        #   m3: [B,256,H/16,W/16]-> concat=512  -> max_pool2d=256   -> lwca4 dim_diy=512
        # =========================================================
        self.cnn_mov = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1), nn.PReLU(),
                nn.AvgPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1), nn.PReLU(),
                nn.AvgPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1), nn.PReLU(),
                nn.AvgPool2d(2, 2),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1), nn.PReLU(),
                nn.AvgPool2d(2, 2),
            ),
        ])
        # 每个尺度: concat(mov, fix) -> 通道翻倍 -> projection 到 LWCA 期望的 dim_diy
        self.cnn_proj0 = nn.Conv2d(64,  64,   1)   # H,   W  -> [B,64,H,W]
        self.cnn_proj1 = nn.Conv2d(128, 128, 1)   # H/2, W/2 -> [B,128,H/2,W/2]
        self.cnn_proj2 = nn.Conv2d(256, 256, 1)   # H/4, W/4 -> [B,256,H/4,W/4]
        self.cnn_proj3 = nn.Conv2d(512, 512, 1)   # H/8, W/8 -> [B,512,H/8,W/8]

        self.up0 = Decoder.DecoderBlock(512, 256, skip_channels=256, use_batchnorm=False)
        self.up1 = Decoder.DecoderBlock(256, 128, skip_channels=128, use_batchnorm=False)
        self.up2 = Decoder.DecoderBlock(128, 64, skip_channels=64, use_batchnorm=False)
        self.up3 = Decoder.DecoderBlock(64, 32, skip_channels=32, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        # [Ablation] Final fusion conv: 替代 DecoderBlock(up3) 处理 x=[64,H,W] + f4=[32,2H,2W]
        self.fusion_conv1 = nn.Conv2d(64 + 32, 32, 3, 1, 1)
        self.fusion_act = nn.PReLU()

        # [Ablation] CNN-only 分支的降维 projection: concat 后通道数减半，输出通道对齐 dim_diy
        self.cnn_proj_lwca0 = nn.Conv2d(128, 64,  1)   # concat(m1,fm1)=128 -> 64  (lwca1 dim_diy=64)
        self.cnn_proj_lwca1 = nn.Conv2d(256, 128, 1)   # concat(m2,fm2)=256 -> 128 (lwca2 dim_diy=128)
        self.cnn_proj_lwca2 = nn.Conv2d(512, 256, 1)   # concat(m3,fm3)=512 -> 256 (lwca3 dim_diy=256)

        self.reg_head = Decoder.RegistrationHead(
            in_channels=32,
            out_channels=2,
            kernel_size=3,
        )

        # =========================================================
        # Image-conditioned Motion Embedding + FiLM
        # ------------------------------------------------------------
        # 替代 Phase-aware FiLM：由 moving/fixed 图像对自动学习 motion
        # embedding，再用该 embedding 调制 reg_head 前的 32 维特征：
        #     feature = feature * (1 + gamma) + beta
        # 其中 gamma / beta 由 motion_embedding 经两层 Linear 产生。
        #
        # 设计要点 (Step 1–5):
        #   - MotionEncoder: 轻量 CNN，输入 cat(moving, fixed)=2 通道，
        #     输出 16 维 motion_embedding。
        #   - 调制通道数 = reg_head 输入通道数 = 32
        #   - 零初始化 gamma/beta: 训练起点等价于无 FiLM (identity)
        #   - use_motion_film=False ⇒ 完全不构造/使用 motion 路径，等价 baseline
        #
        # 兼容性：
        #   - forward 仍接收 phase_id 参数，但 use_motion_film=True 时忽略。
        #   - use_motion_film=False 时仍可传 phase_id（旧行为兼容）。
        # =========================================================
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.PReLU(),
        )
        # gamma / beta 头：零初始化 → FiLM 在训练起点是 identity
        self.motion_gamma_fc = nn.Linear(16, 32)
        self.motion_beta_fc  = nn.Linear(16, 32)
        nn.init.normal_(self.motion_gamma_fc.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.motion_gamma_fc.bias)

        nn.init.normal_(self.motion_beta_fc.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.motion_beta_fc.bias)

        # =========================================================
        # Motion Fusion MLP: 把 image-conditioned motion_embedding 和
        # phase_embedding 融合成最终 motion_code (16 维)，
        # 用它驱动 FiLM (gamma/beta) 并作为 trajectory 一致性约束的目标。
        #
        # 设计动机:
        #   - 之前的 motion_encoder 只看 (moving, fixed) pair，
        #     9 个 phase 之间高度相似（同样的 fixed，不同 phase 的 moving
        #     在呼吸运动上是同一条轨迹），所以 motion_code 几乎一样，
        #     z_jump ≈ 0, z_acc ≈ 0，trajectory loss 失效。
        #   - 现在把 phase_id 也注入，相同 image pair + 不同 phase_id
        #     会得到不同的 motion_code，trajectory 在 z 空间就有了结构。
        # =========================================================
        self.motion_fusion_mlp = nn.Sequential(
            nn.Linear(32, 32),
            nn.PReLU(),
            nn.Linear(32, 16),
            nn.PReLU(),
        )

        # 兼容旧版 phase-aware FiLM: 保留参数但不依赖。
        # - use_motion_film=True  时完全不读 phase_embedding/gamma_fc/beta_fc
        # - use_motion_film=False 时若 forward 仍传入 phase_id（兼容旧调用），
        #   走 phase 路径，与旧实现完全一致
        self.phase_embedding = nn.Embedding(
            num_embeddings=9,
            embedding_dim=16,
        )
        self.gamma_fc = nn.Linear(16, 32)
        self.beta_fc  = nn.Linear(16, 32)

        print(f'[LDMMorph] use_motion_film = {self.use_motion_film}')

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def _cnn_encode(self, img):
        """CNN-only 编码器: [B,1,H,W] -> 4 个尺度特征 (与 Swin 下采样对齐)"""
        f0 = self.cnn_mov[0](img)                          # [B,32,H/2,W/2]
        f1 = self.cnn_mov[1](f0)                           # [B,64,H/4,W/4]
        f2 = self.cnn_mov[2](f1)                           # [B,128,H/8,W/8]
        f3 = self.cnn_mov[3](f2)                           # [B,256,H/16,W/16]
        return f0, f1, f2, f3

    def forward(self, moving_Input, fixed_Input, score1, score2, score3, score4,
                phase_id=None):
        """
        Args:
            moving_Input : [B, 1, H, W]
            fixed_Input  : [B, 1, H, W]
            score1..score4 : LDM 4 个尺度的特征 (与原模型一致)
            phase_id     : [B] long tensor, 取值 0..8.
                           - None     → 完全跳过 FiLM, 与原实现行为一致 (向后兼容)
                           - 提供时  → phase_embedding → FiLM → reg_head 前微调 32 通道特征

        Returns:
            v            : [B, 2, H, W]   registration flow / displacement
            domain_feat  : [B, C, H', W'] swin_fea_8 (供 domain discriminator, 保持不变)
            motion_code  : [B, 16]        仅当 use_motion_film=True 时返回（与 motion_encoder 的 16 维
                                            embedding 对齐），供下游 consistency/周期性 loss；
                                            否则返回 None。
        """
        # motion_code 由后续 FiLM 分支按需生成，此处先占位 None。
        # 这样函数签名仍是 (v, domain_feat)，与老调用解包兼容；
        # 仅当 use_motion_film=True 时会改成 (v, domain_feat, motion_code)。
        motion_code = None
        input_fusion = torch.cat((moving_Input, fixed_Input), dim=1)

        x_s1 = self.avg_pool(input_fusion)
        f4 = self.ec1(x_s1)   # [B,32,256,256]

        swin_fea_4, swin_fea_8, swin_fea_16, swin_fea_32 = self.lwsa(input_fusion)
        # print('swin_fea_4:', swin_fea_4.shape)
        # print('swin_fea_8:', swin_fea_8.shape)
        # print('swin_fea_16:', swin_fea_16.shape)
        # print('swin_fea_32:', swin_fea_32.shape)

        if not self.use_ldm:
            # =========================================================
            # [Ablation] CNN-only 编码: 用 CNN 特征替代 LDM 特征进入 LWCA
            # score0~4 仍然在 forward 中接收（latent loss 继续使用 score0），
            # 但这里不用它们，直接用 CNN 编码的 mov/fix 特征替代
            # =========================================================
            _, m1, m2, m3 = self._cnn_encode(moving_Input)   # [32,256], [64,128], [128,64], [256,32]
            _, fm1, fm2, fm3 = self._cnn_encode(fixed_Input)

            c0 = torch.cat([m1, fm1], dim=1)     # [B,128,128,128]
            c1 = torch.cat([m2, fm2], dim=1)     # [B,256,64,64]
            c2 = torch.cat([m3, fm3], dim=1)     # [B,512,32,32]

            cnn_fea_4  = self.cnn_proj_lwca0(
                F.interpolate(c0, size=swin_fea_4.shape[2:], mode='bilinear', align_corners=False))  # ->[64,128,128]
            cnn_fea_8  = self.cnn_proj_lwca1(c1)   # ->[128,64,64]
            cnn_fea_16 = self.cnn_proj_lwca2(c2)   # ->[256,32,32]
            cnn_fea_32 = F.max_pool2d(c2, 2)      # ->[256,16,16]
        else:
            cnn_fea_4, cnn_fea_8, cnn_fea_16, cnn_fea_32 = (
                self.c1(score1), self.c2(score2), self.c3(score3), self.c4(score4)
            )

        moving_fea_4_cross  = self.lwca1(swin_fea_4,  cnn_fea_4)
        moving_fea_8_cross  = self.lwca2(swin_fea_8,  cnn_fea_8)
        moving_fea_16_cross = self.lwca3(swin_fea_16, cnn_fea_16)
        moving_fea_32_cross = self.lwca4(swin_fea_32, cnn_fea_32)

        fixed_fea_4_cross  = self.lwca1(cnn_fea_4,  swin_fea_4)
        fixed_fea_8_cross  = self.lwca2(cnn_fea_8,  swin_fea_8)
        fixed_fea_16_cross = self.lwca3(cnn_fea_16, swin_fea_16)

        x = self.up0(moving_fea_32_cross, moving_fea_16_cross, fixed_fea_16_cross)  # -> [256,32,32]
        x = self.up1(x, moving_fea_8_cross, fixed_fea_8_cross)                      # -> [128,64,64]
        x = self.up2(x, moving_fea_4_cross, fixed_fea_4_cross)                      # -> [64,128,128]

        if self.use_ldm:
            # LDM 分支: DecoderBlock(up3) 能自动裁剪空间维度的 2x 差异
            x = self.up3(x, f4)      # up -> concat(x,f4=[32,256,256]) -> conv1 -> conv2
            x = self.up(x)           # -> [32,512,512]
        else:
            # CNN-only 分支: 手动处理空间对齐
            x = self.up(x)                                                              # -> [64,256,256]
            x = torch.cat([x, f4], dim=1)                                               # -> [96,256,256]
            x = self.fusion_conv1(x)                                                    # -> [32,256,256]
            x = self.fusion_act(x)
            x = self.up(x)                                                              # -> [32,512,512]

        # =========================================================
        # Conditioning → FiLM (在 reg_head 之前调制 32 维特征)
        # ------------------------------------------------------------
        # 优先级:
        #   1) use_motion_film=True  → 用 motion_encoder(moving, fixed)
        #                                生成 motion_embedding，再走 FiLM。
        #                                （不再读 phase_id）
        #   2) 否则若 phase_id is not None → 走旧版 phase-aware FiLM
        #   3) 否则 → 完全跳过 FiLM，等价旧 baseline
        #
        # 不影响 domain_feat (swin_fea_8) / GRL / 反向域对齐。
        # =========================================================
        film_first_call_logged = getattr(self, '_film_first_call_logged', False)

        if self.use_motion_film:
            motion_input = torch.cat([moving_Input, fixed_Input], dim=1)   # [B,2,H,W]
            image_motion = self.motion_encoder(motion_input)               # [B,16]

            if phase_id is not None:
                # =========================================================
                # 多相位训练：把 phase_embedding 也注入 motion_code，
                # 强制 9 个 phase 之间有差异，避免 motion_code 塌缩成常量。
                # =========================================================
                phase_feat = self.phase_embedding(phase_id)                # [B,16]
                motion_code = self.motion_fusion_mlp(
                    torch.cat([image_motion, phase_feat], dim=1)
                )                                                         # [B,16]
            else:
                # =========================================================
                # 兼容路径（旧 pairwise 脚本调用时 phase_id=None）：
                # 直接用 image_motion，保持与旧实现数值一致。
                # =========================================================
                motion_code = image_motion

            gamma = self.motion_gamma_fc(motion_code).unsqueeze(-1).unsqueeze(-1)  # [B,32,1,1]
            beta  = self.motion_beta_fc(motion_code).unsqueeze(-1).unsqueeze(-1)   # [B,32,1,1]
            x = x * (1.0 + gamma) + beta
            # 把最终 motion_code 暴露给下游 multi-phase 训练（consistency / periodic loss）
            if not film_first_call_logged:
                print(f'[LDMMorph] motion_code: {tuple(motion_code.shape)} | '
                      f'gamma: {tuple(gamma.shape)} | beta: {tuple(beta.shape)} | '
                      f'x before reg_head: {tuple(x.shape)}')
                self._film_first_call_logged = True
        elif phase_id is not None:
            # 兼容旧版 phase-aware FiLM（默认仍 ON，仅当 use_motion_film=False
            # 且调用方仍传 phase_id 时生效，行为与上一版完全一致）
            phase_feat = self.phase_embedding(phase_id)        # [B, 16]
            gamma = self.gamma_fc(phase_feat).unsqueeze(-1).unsqueeze(-1)  # [B, 32, 1, 1]
            beta  = self.beta_fc(phase_feat).unsqueeze(-1).unsqueeze(-1)   # [B, 32, 1, 1]
            x = x * (1.0 + gamma) + beta

        v = self.reg_head(x)
        domain_feat = swin_fea_8

        # ===== 兼容性包装 =====
        # 老代码:  v, domain_feat = model(...)
        # 新代码 (use_motion_film=True): v, domain_feat, motion_code = model(...)
        # 用 2 元 / 3 元 tuple 自适应解包，pairwise 训练脚本完全不受影响。
        if self.use_motion_film:
            return v, domain_feat, motion_code   # motion_code: [B, 16]
        return v, domain_feat    # x: [B,128,H,W] intermediate feature for domain discriminator
