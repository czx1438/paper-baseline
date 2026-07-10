# LDM-Morph 论文复现汇报

## 论文概述

**论文标题**: LDM-Morph: Latent Diffusion Model Guided Deformable Image Registration

**作者**: Jiong Wu, Kuang Gong

**发表**: arXiv preprint arXiv:2411.15426 (2024)

---

## 一、研究背景与问题

### 1.1 医学图像配准的重要性

医学图像配准是医学影像分析中的基础任务，旨在找到两幅图像之间的空间对应关系。在心脏影像分析中，配准技术可用于：
- 追踪心脏运动变化
- 融合不同模态的医学图像
- 辅助诊断心脏疾病
- 术前规划与术后评估

### 1.2 现有方法的局限性

当前基于深度学习的可变形图像配准方法主要采用：

| 方法类型 | 代表模型 | 主要问题 |
|---------|---------|----------|
| CNN方法 | VoxelMorph | 缺乏语义信息，感受野受限 |
| Transformer方法 | TransMorph | 计算复杂度高，忽略高层语义特征 |

**核心问题**：
1. **语义信息缺失**: 传统方法学习的特征缺乏语义级别的对应信息
2. **损失函数局限**: 相似性度量仅在像素空间进行，忽略了解剖结构匹配
3. **拓扑保持问题**: 可能导致形变场折叠(folding)，破坏图像拓扑结构

---

## 二、核心创新点及代码详解

---

## 创新点一：潜在扩散模型(LDM)引导的语义特征提取

### 2.1.1 原理说明

传统方法直接从图像学习特征，这些特征往往是低层次的纹理和边缘信息，缺乏对解剖结构的语义理解。

论文的核心思想是利用**预训练的潜在扩散模型(Latent Diffusion Model)**来提取丰富的语义特征：

1. **为什么LDM能提取语义特征？**
   - LDM在大规模图像数据集上训练，学会了理解图像的高层语义
   - 扩散模型的去噪过程隐式地编码了图像的结构信息
   - UNet编码器-解码器结构天然具有多尺度特征提取能力

2. **为什么要在潜在空间提取特征？**
   - 潜在空间维度更低，计算更高效
   - 潜在表示更紧凑，语义信息更集中
   - 便于进行特征级别的相似性度量

### 2.1.2 代码逐行详解

#### 第一步：加载预训练的LDM模型

```python
# 文件: train.py 第73-91行
def load_model_from_config(config, sd):
    """
    从配置文件和权重字典加载LDM模型
    config: OmegaConf配置对象，包含模型结构定义
    sd: state_dict，包含预训练权重
    """
    # instantiate_from_config: 根据config中的配置实例化模型对象
    # 这个函数会创建LDM模型的所有组件：AutoEncoder、UNet、扩散调度器等
    model = instantiate_from_config(config)
    
    # load_state_dict: 将预训练权重加载到模型中
    # strict=False表示允许部分权重不匹配（如新增的层）
    model.load_state_dict(sd, strict=False)
    
    # 将模型移到GPU上，因为后续计算需要GPU加速
    model.cuda()
    
    # 设置为评估模式
    # 作用：禁用dropout、BatchNorm使用全局统计量等
    # 这样可以保证推理的一致性
    model.eval()
    
    return model
```

#### 第二步：完整训练加载模型

```python
# 文件: train.py 第80-91行
def load_model(config, ckpt, gpu, eval_mode):
    """
    加载完整模型的包装函数
    config: 配置文件路径列表
    ckpt: 预训练权重路径
    gpu: 是否使用GPU
    eval_mode: 是否评估模式
    """
    if ckpt:
        print(f"Loading model from {ckpt}")
        # torch.load: 从磁盘加载预训练权重
        # map_location="cpu": 即使有GPU也先加载到CPU，再根据需要移动
        pl_sd = torch.load(ckpt, map_location="cpu")
        
        # global_step: 从checkpoint中获取训练步数，用于日志记录
        global_step = pl_sd["global_step"]
    else:
        # 如果没有checkpoint，创建空的state_dict
        pl_sd = {"state_dict": None}
        global_step = None
    
    # 调用上面的加载函数
    model = load_model_from_config(config.model, pl_sd["state_dict"])
    return model, global_step
```

#### 第三步：图像编码到潜在空间

```python
# 文件: train.py 第182-183行
# 这两行是特征提取的核心

# encode_first_stage: LDM的编码器，将图像从像素空间压缩到潜在空间
# 输入X: [B, 1, H, W] 原始图像 (B=batch, H/W=图像尺寸)
# 输出: [B, 4, H/8, W/8] 潜在表示 (通道数=4，空间尺寸缩小8倍)

# get_first_stage_encoding: 对编码结果进行量化/归一化处理
# 作用：将编码器输出转换为潜在空间的标准表示形式
# .detach(): 阻断梯度反传，因为LDM特征是作为固定输入，不需要更新
mov_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(X)).detach()

# 同样的处理应用于参考图像Y
fix_z = ldm_model.get_first_stage_encoding(ldm_model.encode_first_stage(Y)).detach()
```

**为什么要detach()？**
```
mov_z是LDM提取的特征，作为LDMMorph的输入
如果不断开梯度，每次反向传播时：
  loss → LDMMorph参数更新 ✓ (这是我们想要的)
       → LDM参数更新 ✗ (LDM应该保持固定！)

.detach()后的计算图：
  X → encode → detach → mov_z → LDMMorph → loss
                              ↑
                         梯度到此截断，不会流向LDM
```

#### 第四步：扩散过程添加噪声

```python
# 文件: train.py 第185-188行

# default: 如果noise为None，则生成随机噪声
# torch.randn_like(mov_z): 生成与mov_z形状相同的标准正态分布噪声
# 作用：作为扩散过程的随机扰动
noise = None
noise = default(noise, lambda: torch.randn_like(mov_z))

# q_sample: 扩散模型的加噪过程
# x_start: 原始干净图像 mov_z
# t: 时间步 torch.tensor([t_enc]) 即 t=1
# noise: 添加的噪声
# 公式: x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
# 其中ᾱ_t是关于时间步的衰减系数
x_noisy = ldm_model.q_sample(x_start=mov_z, t=torch.tensor([t_enc]).cuda(), noise=noise)

# 对参考图像做同样的加噪处理，使用相同的噪声
# 作用：确保两幅图像受到相同程度的扰动，便于后续特征对比
y_noisy = ldm_model.q_sample(x_start=fix_z, t=torch.tensor([t_enc]).cuda(), noise=noise)
```

**为什么t=1而不是更大的值？**
```
扩散模型的去噪过程: t从大到小
t=1000 (完全噪声) → ... → t=1 (轻微噪声) → t=0 (原始图像)

选择t=1的原因：
1. 噪声很少，图像结构基本保留
2. LDM在轻微噪声扰动下提取的特征更具判别性
3. 计算效率高，不需要完整去噪过程

t较大时的特征可能过于抽象，不利于精确配准
```

#### 第五步：UNet去噪并提取多尺度特征

```python
# 文件: train.py 第190-191行

# apply_model: LDM的去噪UNet
# 输入: 加噪图像 x_noisy, 时间步 t
# 输出: 预测的噪声和中间特征 (通过return_ids=True控制)

# cond=None: 无条件生成，因为我们不需要文本等条件信息
# return_ids=True: 返回UNet中间层的特征，用于配准网络
outx = ldm_model.apply_model(x_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)

# 同理处理参考图像的加噪版本
outy = ldm_model.apply_model(y_noisy, t=torch.tensor([t_enc]).cuda(), cond=None, return_ids=True)
```

#### 第六步：从UNet输出中提取特定层特征

```python
# 文件: train.py 第193-196行

# outx的结构: (预测噪声, (多层特征列表))
# outx[1]: 特征列表
# outx[1][0]: 第0层的特征 (B, C, H, W)
# outx[1][0][0]: 第0层第0个特征图的第0个元素 (标量，无实际意义)
# 实际上 outx[1][0] 就是第0层的完整特征: [B, C, H, W]

# torch.cat: 沿着通道维度拼接特征
# (浮动图像第0层特征, 参考图像第0层特征)
# 作用：将浮动图像和参考图像的同层特征融合，便于后续比较

# score0: 浅层特征，分辨率高(1/4)，通道数128+192=320
# 包含：outx第0层 + outy第0层 (各取特定通道)
# outx[1][0][0] 和 outx[1][0][2]: 来自outx的某些通道
# outy[1][0][0] 和 outy[1][0][2]: 来自outy的对应通道
score0 = torch.cat((
    outx[1][0][0],   # 浮动图 UNet第0层特征的一部分
    outx[1][0][2],   # 浮动图 UNet第0层特征的另一部分
    outy[1][0][0],   # 参考图 UNet第0层特征的一部分
    outy[1][0][2]    # 参考图 UNet第0层特征的另一部分
), dim=1)            # dim=1表示在通道维度拼接

# score1: 中层特征，分辨率1/2，通道数192+320=512
# 同样拼接浮动图和参考图的多层特征
score1 = torch.cat((
    outx[1][0][3],   # 浮动图 UNet第1层特征
    outx[1][0][5],
    outy[1][0][3],   # 参考图 UNet第1层特征
    outy[1][0][5]
), dim=1)

# score2: 深层特征，分辨率1/4，通道数320+448=768
score2 = torch.cat((
    outx[1][0][6],   # 浮动图 UNet第2层特征
    outx[1][0][8],
    outy[1][0][6],   # 参考图 UNet第2层特征
    outy[1][0][8]
), dim=1)

# score3: 最深层特征，分辨率1/8，通道数448+512=960
# 包含最抽象的语义信息
score3 = torch.cat((
    outx[1][0][9],   # 浮动图 UNet第3层特征
    outx[1][0][11],
    outy[1][0][9],   # 参考图 UNet第3层特征
    outy[1][0][11]
), dim=1)
```

**为什么要取特定的通道索引(0,2,3,5,6,8,9,11)？**
```
这些是LDM的UNet中 encoder-decoder skip connection 的特征输出
索引含义：
- 0,2: encoder第1层和decoder第1层的特征 (浅层)
- 3,5: encoder第2层和decoder第2层的特征 (中层)
- 6,8: encoder第3层和decoder第3层的特征 (深层)
- 9,11: encoder第4层和decoder第4层的特征 (最深层)

每个跳跃连接提供不同尺度的信息：
浅层(0,2): 更多纹理细节
深层(9,11): 更多语义结构
```

#### 特征维度汇总表

| 特征名 | 来源层 | 通道数计算 | 输出尺寸 | 语义层级 |
|-------|-------|-----------|---------|---------|
| score0 | UNet层0+2 | 128×2 + 192×2 = 640 | 1/4原图 | 纹理/边缘 |
| score1 | UNet层3+5 | 192×2 + 320×2 = 1024 | 1/8原图 | 局部结构 |
| score2 | UNet层6+8 | 320×2 + 448×2 = 1536 | 1/16原图 | 区域语义 |
| score3 | UNet层9+11 | 448×2 + 512×2 = 1920 | 1/32原图 | 全局语义 |

---

## 创新点二：局部窗口自注意力模块(LWSA)

### 2.2.1 原理说明

传统的自注意力机制需要计算所有像素之间的注意力分数，计算复杂度为 O(H²W²)，对于高分辨率医学图像来说不可接受。

LWSA基于**Swin-Transformer**的移位窗口机制：

1. **局部窗口注意力**: 将图像划分为不重叠的窗口，在窗口内计算注意力
2. **移位窗口机制(SW-MSA)**: 通过周期性移位实现跨窗口信息交互
3. **多尺度输出**: 通过Patch Merging层逐步降低分辨率，提取多尺度特征

### 2.2.2 代码逐行详解

#### LWSA模块整体结构

```python
# 文件: TransModels/LWSA.py 第22-71行
class LWSA(nn.Module):
    """
    Local Window Self-Attention (LWSA) 模块
    作用：提取多尺度Swin-Transformer特征
    输入: 融合后的浮动图像和参考图像 [B, 2, H, W]
    输出: 4个尺度的特征 [f3, f2, f1, f4] 对应分辨率1/4, 1/8, 1/16, 1/32
    """
    def __init__(self, config, in_channel=1):
        super(LWSA, self).__init__()

        # 从config获取参数
        if_convskip = config.if_convskip    # 是否使用卷积skip连接
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip  # 是否使用Transformer skip连接
        self.if_transskip = if_transskip

        embed_dim = config.embed_dim        # 嵌入维度

        # 创建Swin-Transformer backbone
        self.transformer = basic.SwinTransformer(
            patch_size=config.patch_size,      # Patch大小，默认4
            in_chans=config.in_chans,         # 输入通道数
            embed_dim=config.embed_dim,       # 嵌入维度 36
            depths=config.depths,             # 每层深度 [2, 2, 6, 2]
            num_heads=config.num_heads,       # 注意力头数 [3, 6, 12, 24]
            window_size=config.window_size,   # 窗口大小，默认8
            mlp_ratio=config.mlp_ratio,       # MLP隐藏层比率 4.0
            qkv_bias=config.qkv_bias,        # QKV偏置 True
            drop_rate=config.drop_rate,       # Dropout比率 0.0
            drop_path_rate=config.drop_path_rate,  # 随机深度 0.2
            ape=config.ape,                   # 绝对位置编码 False
            patch_norm=config.patch_norm,    # Patch归一化 True
            use_checkpoint=config.use_checkpoint,  # 梯度checkpoint False
            out_indices=config.out_indices    # 输出哪些层的特征 (0,1,2,3)
        )

        # 卷积层：将原始2通道输入转换为embed_dim/2通道
        self.c1 = Conv2dReLU.Conv2dReLU(in_channel, embed_dim//2, 3, 1, use_batchnorm=False)
        # 作用：提供浅层卷积特征作为skip connection

        # 注册头通道数，用于后续处理
        self.c2 = Conv2dReLU.Conv2dReLU(in_channel, config.reg_head_chan, 3, 1, use_batchnorm=False)

        # 平均池化：下采样2倍
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
```

#### LWSA前向传播

```python
    def forward(self, x):
        source = x  # 保存原始输入

        if self.if_convskip:
            # 复制输入用于skip连接
            x_s0 = x.clone()

            # 池化下采样
            x_s1 = self.avg_pool(x)
            # 作用：将256×256下采样到128×128

            # 卷积处理
            f4 = self.c1(x_s1)
            # 输出: [B, embed_dim//2, H/2, W/2]

            # 原始分辨率的卷积特征
            f5 = self.c2(x_s0)
            # 输出: [B, reg_head_chan, H, W]
        else:
            # 如果不使用skip连接，设为None
            f4 = None
            f5 = None

        # Swin-Transformer特征提取
        out_feats = self.transformer(x)
        # 输出: 4个元组的列表 [(B,C1,H1,W1), (B,C2,H2,W2), (B,C3,H3,W3), (B,C4,H4,W4)]

        if self.if_transskip:
            # 提取中间层的Transformer特征用于skip连接
            f1 = out_feats[-2]  # 第2层输出 (1/16分辨率)
            f2 = out_feats[-3]  # 第1层输出 (1/8分辨率)
            f3 = out_feats[-4]  # 第0层输出 (1/4分辨率)
        else:
            f1 = None
            f2 = None
            f3 = None

        # 返回4个尺度的特征 (从浅到深)
        return f3, f2, f1, out_feats[-1]
        # f3: 1/4分辨率, f2: 1/8, f1: 1/16, out_feats[-1]: 1/32
```

#### Swin-Transformer核心：窗口注意力

```python
# 文件: TransModels/basic_LWSA.py 第81-161行
class WindowAttention(nn.Module):
    """
    窗口注意力机制
    核心思想：在不重叠的局部窗口内计算自注意力
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, ...):
        super().__init__()
        self.dim = dim                    # 输入维度
        self.window_size = window_size     # 窗口大小，如8×8
        self.num_heads = num_heads         # 注意力头数

        # 每个头的维度 = 总维度 / 头数
        head_dim = dim // num_heads
        # 缩放因子：1/√(head_dim)
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置偏置表
        # 形状: [(2*window_size-1)², num_heads]
        # 作用：编码像素之间的相对位置关系
        # 这是Swin-Transformer的关键创新之一
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # 计算相对位置索引
        # 作用：为每个窗口内的像素对分配相对位置坐标
        coords_h = torch.arange(self.window_size[0])  # [0,1,2,...,window_size-1]
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # [2, Wh*Ww]

        # 计算所有像素对的相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # 形状: [2, Wh*Ww, Wh*Ww]
        # 作用：relative_coords[d,i,j] = coords[d,i] - coords[d,j]

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # 形状: [Wh*Ww, Wh*Ww, 2]

        # 移位到非负范围
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # 作用：使索引从0开始，便于查表

        # 压缩为1D索引
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        # 作用：将2D相对坐标转换为1D索引

        # 注册为buffer（不参与梯度计算，但会随模型保存/移动）
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV线性变换
        # 将输入特征转换为Query, Key, Value
        # dim → 3*dim (Q,K,V各一份)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout
        self.proj = nn.Linear(dim, dim)         # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化相对位置偏置
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)  # softmax在最后一维
```

#### 窗口注意力的前向计算

```python
    def forward(self, x, mask=None):
        """
        前向计算
        x: [num_windows*B, N, C]  N=window_size²
        mask: 注意力掩码，用于SW-MSA
        """
        B_, N, C = x.shape
        # B_: 窗口数量×批量大小
        # N: 每个窗口的像素数 (如64 for 8×8 window)
        # C: 特征维度

        # QKV变换并分割
        # [num_windows*B, N, C] → [num_windows*B, N, 3, num_heads, head_dim]
        # → [3, num_windows*B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q,k,v: [num_windows*B, num_heads, N, head_dim]

        # 缩放Query
        q = q * self.scale
        # 作用：防止点积过大导致softmax梯度消失

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1))
        # q @ k^T: [num_windows*B, num_heads, N, N]
        # 作用：每个头计算所有位置的相似度

        # 添加相对位置偏置
        # 通过查表获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        )
        # 形状: [N, N, num_heads]

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 形状: [num_heads, N, N]

        attn = attn + relative_position_bias.unsqueeze(0)
        # 作用：加入位置信息，让模型感知像素间的距离

        # 应用掩码（如果是SW-MSA）
        if mask is not None:
            nW = mask.shape[0]
            # 重塑注意力分数以应用掩码
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # Softmax归一化
        attn = self.softmax(attn)

        # Dropout
        attn = self.attn_drop(attn)

        # 注意力加权聚合
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # 形状变化: [B,H,N,D]@[B,H,N,D] → [B,H,N,D] → [B,N,H,D] → [B,N,C]

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
```

**为什么需要相对位置偏置？**
```
普通自注意力是位置无关的：attn(i,j) = softmax(q_i · k_j)

但图像中的空间关系很重要：
- 像素i和i+1的关系 ≠ 像素i和i+100的关系

相对位置偏置：
attn(i,j) = softmax(q_i · k_j + bias(i-j))

bias是一个可学习的表，根据相对位置(i-j)查表获取
这样注意力分数就包含了位置信息
```

#### Swin-Transformer Block：移位窗口机制

```python
# 文件: TransModels/basic_LWSA.py 第164-263行
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block
    关键特性：交替使用常规窗口注意力和移位窗口注意力
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, ...):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size  # 0=常规窗口, window//2=移位窗口

        self.norm1 = norm_layer(dim)  # LayerNorm
        self.attn = WindowAttention(...)  # 窗口注意力

        # DropPath/Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, ...)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        # L = H * W

        # 残差连接
        shortcut = x

        # LayerNorm
        x = self.norm1(x)

        # 重塑为2D图像格式
        x = x.view(B, H, W, C)
        # 形状: [B, H, W, C]

        # Padding到window_size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        # 作用：确保图像尺寸能被window_size整除

        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # 填充0在右边和下边

        _, Hp, Wp, _ = x.shape
        # Hp, Wp是填充后的尺寸

        # 循环移位 (Shifted Window)
        if self.shift_size > 0:
            # 沿H和W方向反向移位
            # 移位大小为shift_size（通常为window_size//2）
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # 作用：创建跨窗口的信息交互

            attn_mask = mask_matrix  # 注意力掩码
        else:
            shifted_x = x
            attn_mask = None

        # 窗口分区
        # [B, Hp, Wp, C] → [num_windows*B, window_size, window_size, C]
        x_windows = window_partition(shifted_x, self.window_size)

        # 展平窗口
        # [num_windows*B, window_size, window_size, C] → [num_windows*B, window_size², C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 窗口注意力计算
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # 输出: [num_windows*B, window_size², C]

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        # 作用：将窗口重新合并为完整特征图

        # 反向移位，恢复原始位置
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # 去除padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # 恢复为序列格式
        x = x.view(B, H * W, C)

        # FFN残差
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
```

**移位窗口机制图解：**
```
常规窗口注意力 (shift_size=0):
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │
├───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┼───┤
│ 9 │10 │11 │12 │
├───┼───┼───┼───┤
│13 │14 │15 │16 │
└───┴───┴───┴───┘
每个窗口独立计算注意力，无跨窗口交互

移位窗口注意力 (shift_size=window_size//2):
┌───┬───┬───┬───┐
│ 4 │ 1 │ 2 │ 3 │ ← 循环移位
├───┼───┼───┼───┤
│ 8 │ 5 │ 6 │ 7 │
├───┼───┼───┼───┤
│12 │ 9 │10 │11 │
├───┼───┼───┼───┤
│16 │13 │14 │15 │
└───┴───┴───┴───┘
窗口边界发生变化，原来不同窗口的像素现在可以交互

通过交替使用这两种注意力：
- 层1: 常规窗口
- 层2: 移位窗口
- 层3: 常规窗口
- ...

最终实现了类似全局注意力的效果，但计算复杂度更低
```

---

## 创新点三：局部全局交叉注意力模块(LWCA)

### 2.3.1 原理说明

LWCA是LDM-Morph的核心创新，用于融合两种不同来源的特征：

1. **Query来源 (Swin特征)**: 提供局部结构信息和位置关系
2. **Key/Value来源 (LDM特征)**: 提供丰富的语义信息和解剖结构

通过交叉注意力机制，实现局部-全局特征的深度交互。

### 2.3.2 代码逐行详解

#### LWCA模块定义

```python
# 文件: TransModels/LWCA.py 第21-45行
class LWCA(nn.Module):
    """
    Local Window Cross Attention (LWCA) 模块
    作用：融合Swin-Transformer特征（Query）和LDM特征（Key/Value）
    """
    def __init__(self, config, dim_diy):
        super(LWCA, self).__init__()

        # 创建跨域Swin-Transformer
        # 注意：这里使用basic_LWCA中的SwinTransformer
        # 与LWSA中的版本略有不同（支持双输入）
        self.transformer = basic.SwinTransformer(
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            ape=config.ape,
            spe=config.spe,              # 空间嵌入
            patch_norm=config.patch_norm,
            use_checkpoint=config.use_checkpoint,
            out_indices=config.out_indices,
            pat_merg_rf=config.pat_merg_rf,
            dim_diy=dim_diy              # 自定义输出维度
        )

    def forward(self, x, y):
        """
        x: Swin特征 [B, C, H, W] - 作为Query
        y: LDM特征 [B, C', H, W] - 作为Key和Value
        """
        # 调用transformer进行跨域注意力计算
        moving_fea_cross = self.transformer(x, y)
        return moving_fea_cross
```

#### 跨域注意力计算

```python
# 文件: TransModels/basic_LWCA.py 第111-143行
class WindowAttention(nn.Module):
    """
    跨域窗口注意力
    与自注意力的区别：Query和Key/Value来自不同的输入
    """
    def forward(self, x, y, mask=None):
        """
        x: Query来源 [num_windows*B, N, C]
        y: Key/Value来源 [num_windows*B, N, C]
        mask: 注意力掩码
        """
        B_, N, C = x.shape

        # 对x进行QKV变换
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(y).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 注意：x和y分别进行独立的QKV变换

        q = qkv2[0]  # Query来自x
        k, v = qkv[1], qkv[2]  # Key和Value来自y
        # 关键点：Query≠Key/Value，这是跨域注意力的核心

        # 缩放
        q = q * self.scale

        # 计算注意力分数
        # Query来自Swin特征 → 包含局部结构信息
        # Key来自LDM特征 → 包含丰富的语义信息
        attn = (q @ k.transpose(-2, -1))
        # 结果：每个Swin特征位置，对所有LDM特征的注意力权重

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(...)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Softmax
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 注意力加权Value
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # 作用：每个Swin特征位置，根据与LDM特征的相似度，聚合语义信息

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
```

**跨域注意力图解：**
```
Swin特征 (Query)          LDM特征 (Key/Value)
┌─────────────┐         ┌─────────────┐
│ 局部结构    │         │ 语义分割    │
│ 边缘信息    │         │ 解剖结构    │
│ 纹理细节    │         │ 器官边界    │
└──────┬──────┘         └──────┬──────┘
       │                      │
       │ Query                │ Key, Value
       │                      │
       ▼                      ▼
┌─────────────────────────────────────┐
│         交叉注意力计算               │
│  Q · K^T → 相似度矩阵 → softmax    │
│  A · V → 加权聚合                  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│ 融合特征     │
│ 局部+语义    │
└─────────────┘
```

#### LWCA在网络中的使用

```python
# 文件: TransModels/LDMMorph.py 第64-92行

def forward(self, moving_Input, fixed_Input, score1, score2, score3, score4):
    """
    完整前向传播
    moving_Input: 浮动图像
    fixed_Input: 参考图像
    score1-4: LDM提取的多尺度特征
    """

    # 1. 融合浮动图像和参考图像
    input_fusion = torch.cat((moving_Input, fixed_Input), dim=1)
    # 作用：[B,1,H,W] + [B,1,H,W] → [B,2,H,W]

    # 2. 下采样 + 浅层卷积
    x_s1 = self.avg_pool(input_fusion)  # [B,2,H/2,W/2]
    f4 = self.ec1(x_s1)  # [B,16,H/2,W/2] - embed_dim//2=16

    # 3. LWSA提取Swin多尺度特征
    swin_fea_4, swin_fea_8, swin_fea_16, swin_fea_32 = self.lwsa(input_fusion)

    # 4. CNN编码器处理LDM特征
    cnn_fea_4 = self.c1(score1)   # [B, 64, H/4, W/4]
    cnn_fea_8 = self.c2(score2)   # [B, 128, H/8, W/8]
    cnn_fea_16 = self.c3(score3)  # [B, 256, H/16, W/16]
    cnn_fea_32 = self.c4(score4)  # [B, 512, H/32, W/32]
    # 作用：将LDM特征的通道数调整到与Swin特征对齐

    # 5. LWCA交叉融合 - 浮动图像特征
    # Q = Swin特征, K/V = LDM特征
    # 将LDM的语义信息注入到Swin的局部结构中
    moving_fea_4_cross = self.lwca1(swin_fea_4, cnn_fea_4)
    moving_fea_8_cross = self.lwca2(swin_fea_8, cnn_fea_8)
    moving_fea_16_cross = self.lwca3(swin_fea_16, cnn_fea_16)
    moving_fea_32_cross = self.lwca4(swin_fea_32, cnn_fea_32)

    # 6. LWCA交叉融合 - 参考图像特征
    # 注意：Query和Key/Value的角色互换
    # Q = LDM特征, K/V = Swin特征
    # 作用：增强LDM特征的局部定位能力
    fixed_fea_4_cross = self.lwca1(cnn_fea_4, swin_fea_4)
    fixed_fea_8_cross = self.lwca2(cnn_fea_8, swin_fea_8)
    fixed_fea_16_cross = self.lwca3(cnn_fea_16, swin_fea_16)

    # 7. U-Net解码器 + 形变场生成
    x = self.up0(moving_fea_32_cross, moving_fea_16_cross, fixed_fea_16_cross)
    x = self.up1(x, moving_fea_8_cross, fixed_fea_8_cross)
    x = self.up2(x, moving_fea_4_cross, fixed_fea_4_cross)
    x = self.up3(x, f4)  # 最后一层使用浅层卷积特征
    x = self.up(x)  # 上采样回原始分辨率

    v = self.reg_head(x)  # 生成2通道形变场

    return v
```

---

## 创新点四：分层度量损失函数

### 2.4.1 原理说明

传统配准方法只使用像素空间的相似性度量（如MSE、NCC），忽略了高层语义对应。LDM-Morph提出分层度量，同时考虑：

1. **像素空间相似性**: 保证像素级对齐
2. **潜在空间相似性**: 增强语义级别的匹配
3. **平滑正则化**: 防止形变场折叠

### 2.4.2 代码逐行详解

#### 分层损失计算

```python
# 文件: train.py 第204-206行

# β参数控制两种损失的权重
# β=0.6表示：60%像素损失 + 40%潜在空间损失
loss1 = beta * loss_similarity(Y, X_Y) + (1-beta) * loss_similarity(mov_z, fix_z)

"""
loss1分解：
1. beta * loss_similarity(Y, X_Y)
   - Y: 参考图像 [B, 1, H, W]
   - X_Y: 变形后的浮动图像 [B, 1, H, W]
   - 作用：像素空间的相似性，最小化两幅图像的像素差异
   - 优点：保证像素级对齐，细节保留

2. (1-beta) * loss_similarity(mov_z, fix_z)
   - mov_z: 变形后图像X_Y在LDM潜在空间的表示 [B, 4, H/8, W/8]
   - fix_z: 参考图像Y在LDM潜在空间的表示 [B, 4, H/8, W/8]
   - 作用：潜在空间的相似性，增强语义匹配
   - 优点：利用LDM的语义理解能力，保证解剖结构对应
"""

# 平滑正则化损失
loss2 = loss_smooth(D_f_xy)
"""
D_f_xy: 预测的形变场 [B, 2, H, W]
作用：惩罚形变场的梯度，防止相邻像素的位移差异过大
"""

# 总损失
loss = loss1 + smooth * loss2
"""
smooth: 平滑损失的权重，默认0.01
作用：平衡相似性优化和形变场平滑性
"""
```

#### MSE损失实现

```python
# 文件: utils/utils.py 第80-86行

class MSE:
    """
    均方误差损失
    用于衡量两幅图像的相似性
    """
    def loss(self, y_true, y_pred):
        """
        y_true: 参考图像
        y_pred: 变形后的图像
        """
        return torch.mean((y_true - y_pred) ** 2)
        """
        计算过程：
        1. y_true - y_pred: 逐像素相减 [B, C, H, W]
        2. (diff) ** 2: 平方 [B, C, H, W]
        3. torch.mean: 求平均得到标量

        特点：
        - 对大差异惩罚更大 (平方项)
        - 对小差异惩罚较小
        - 适用于灰度图像的相似性度量
        """
```

#### 平滑正则化损失

```python
# 文件: utils/utils.py 第63-67行

def smoothloss(y_pred):
    """
    平滑正则化损失
    作用：惩罚形变场的一阶导数（梯度），保证形变场平滑
    防止形变场出现折叠、撕裂等拓扑错误
    """
    h2, w2 = y_pred.shape[-2:]
    # 获取空间尺寸

    # 计算水平方向的梯度
    # y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]
    #   = 相邻像素的位移差 [B, 2, H-1, W]
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / (2 * h2)
    # 除以2*h2是归一化，将梯度缩放到合理范围

    # 计算垂直方向的梯度
    # y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]
    #   = 相邻像素的位移差 [B, 2, H, W-1]
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / (2 * w2)
    # 同样归一化

    # L2范数惩罚
    return (torch.mean(dx * dx) + torch.mean(dz * dz)) / 2.0
    """
    为什么用L2范数 (平方) 而不是L1？
    - L2对大梯度惩罚更大
    - 梯度更小时优化更稳定
    - 但L1可能产生更稀疏的解

    为什么除以2？
    - 求平均后除以2，等价于对两个方向的损失取平均
    """

smoothloss数学解释：
设形变场为 u(x,y) = (u₁(x,y), u₂(x,y))

梯度 ∇u 表示相邻像素位移的变化率

平滑损失 = E[(∂u₁/∂x)²] + E[(∂u₁/∂y)²] + E[(∂u₂/∂x)²] + E[(∂u₂/∂y)²]

当梯度为0时，形变场完全平滑（刚性变换）
当梯度很大时，形变场剧烈变化（可能产生折叠）

通过最小化这个损失，形变场趋向于平滑连续
```

**平滑正则化效果对比：**
```
无平滑正则化（可能产生折叠）:
┌─────────────────────────┐
│  ↑↓↑↓↑↓↑↓↑↓              │ ← 形变场不规则
│  ↓↑↓↑↓↑↓↑↓↑              │
│  ↑↓↑↓↑↓↑↓↑↓              │
│  折叠区域!               │
└─────────────────────────┘

有平滑正则化（平滑连续）:
┌─────────────────────────┐
│  ↗↗↗↗↗↗↗↗↗              │ ← 形变场平滑
│  ↗↗↗↗↗↗↗↗↗              │
│  ↗↗↗↗↗↗↗↗↗              │
│  无折叠，拓扑保持         │
└─────────────────────────┘
```

#### 完整训练流程中的损失计算

```python
# 文件: train.py 第198-209行

# 1. 前向传播获取形变场
D_f_xy = model(X, Y, score0, score1, score2, score3)
# D_f_xy: [B, 2, H, W] 预测的形变场

# 2. 空间变换，对浮动图像进行变形
_, X_Y = transform(X, D_f_xy.permute(0, 2, 3, 1), mod='nearest')
# transform: STN空间变换函数
# D_f_xy.permute: [B,2,H,W] → [B,H,W,2] (grid_sample需要的格式)
# mod='nearest': 使用最近邻插值（适用于分割掩码）
# X_Y: 变形后的浮动图像

# 3. 重新计算变形后图像的LDM特征
mov_z = ldm_model.get_first_stage_encoding(
    ldm_model.encode_first_stage(X_Y)
).detach()
# 为什么重新计算？
# 因为X_Y是变形后的图像，其LDM特征应该更接近fix_z

# 4. 计算分层损失
loss1 = beta * loss_similarity(Y, X_Y) + (1-beta) * loss_similarity(mov_z, fix_z)
# 第一项：像素空间MSE
# 第二项：潜在空间MSE

loss2 = loss_smooth(D_f_xy)
# 平滑正则化

loss = loss1 + smooth * loss2
# 总损失

# 5. 反向传播
optimizer.zero_grad()
loss.backward()      # 计算梯度
optimizer.step()     # 更新参数
```

#### Dice系数评估

```python
# 文件: train.py 第93-107行

def dice(pred1, truth1):
    """
    计算Dice系数，评估配准质量
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    取值范围 [0, 1]，1表示完美匹配
    """
    # 根据数据集选择标签类别
    if datapath == 'acdc':
        VOI_lbls = [2, 3]  # ACDC有标签2（左心室）、3（右心室）
    else:
        VOI_lbls = [1]     # 其他数据集只有标签1（心肌）

    dice_all = np.zeros(len(VOI_lbls))  # 存储每个类别的Dice
    index = 0

    for k in VOI_lbls:
        # 布尔掩码
        truth = truth1 == k  # 参考图像中属于类别k的像素
        pred = pred1 == k    # 预测图像中属于类别k的像素

        # 交集
        intersection = np.sum(pred * truth) * 2.0
        # pred * truth: 逐元素相与，只有同时为True才为1
        # np.sum: 计算True的个数
        # *2.0: 为Dice公式的2倍

        # Dice系数
        dice_all[index] = intersection / (np.sum(pred) + np.sum(truth))
        # 分母：两幅图像中类别k的像素总数之和

        index += 1

    return np.mean(dice_all)
    # 返回所有类别的平均Dice
```

**Dice系数图解：**
```
参考图像分割          预测分割          交集
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │
│   ████████  │    │  ██████████ │    │   ████      │
│   ████████  │    │  ██████████ │    │   ████      │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘

交集像素 = 4 × 4 = 16 (假设)
参考图像像素 = 4 × 8 = 32
预测图像像素 = 5 × 8 = 40

Dice = 2 × 16 / (32 + 40) = 32 / 72 = 0.444

Dice = 1.0: 完全重合
Dice = 0.0: 完全不重合
```

---

## 三、网络架构详解

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        LDM-Morph 整体架构                        │
└─────────────────────────────────────────────────────────────────┘

输入阶段:
┌──────────────────────┐     ┌──────────────────────┐
│   浮动图像 X [1,256²] │     │   参考图像 Y [1,256²] │
└──────────┬───────────┘     └──────────┬───────────┘
           │                           │
           │            ┌──────────────┴──────────────┐
           │            │                             │
           │            ▼                             ▼
           │     ┌─────────────┐              ┌─────────────┐
           │     │    LDM      │              │    LDM      │
           │     │   编码器    │              │   编码器    │
           │     └──────┬──────┘              └──────┬──────┘
           │            │                             │
           │            ▼                             ▼
           │     ┌─────────────┐              ┌─────────────┐
           │     │ 潜在空间    │              │ 潜在空间    │
           │     │ mov_z      │              │ fix_z      │
           │     └──────┬──────┘              └──────┬──────┘
           │            │                             │
           │            └──────────────┬──────────────┘
           │                           │ 相同噪声扰动
           │                           ▼
           │                  ┌─────────────────┐
           │                  │  UNet去噪 +     │
           │                  │ 特征提取 score0-3│
           │                  └────────┬────────┘
           │                           │
           └──────────┬────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    特征融合阶段                                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  输入融合 [2,256²]                         │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │                                     │
│         ┌────────────────┴────────────────┐                   │
│         │                                 │                   │
│         ▼                                 ▼                   │
│  ┌──────────────┐                  ┌──────────────┐            │
│  │    LWSA      │                  │  CNN编码器   │            │
│  │(Swin特征)    │                  │(LDM特征)      │            │
│  │              │                  │              │            │
│  │f4,f8,f16,f32 │                  │score0-3通道  │            │
│  └──────┬───────┘                  │调整          │            │
│         │                          └──────┬───────┘            │
│         │                                 │                    │
│         └─────────────┬───────────────────┘                    │
│                       │                                          │
│                       ▼                                          │
│              ┌────────────────┐                                │
│              │  LWCA交叉注意力 │                                │
│              │ (4个尺度并行)   │                                │
│              │                │                                │
│              │ moving_cross    │                                │
│              │ fixed_cross     │                                │
│              └───────┬─────────┘                                │
└──────────────────────┼──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    解码输出阶段                                   │
│                                                                  │
│  DecoderBlock × 4: 上采样 + 特征融合                            │
│                       │                                         │
│                       ▼                                         │
│              ┌────────────────┐                                │
│              │ RegistrationHead│                               │
│              │ Conv2d + Softsign│                               │
│              └───────┬─────────┘                                │
│                       │                                          │
│                       ▼                                          │
│              ┌────────────────┐                                 │
│              │ 形变场 D_xy    │                                 │
│              │ [2, 256²]     │                                 │
│              └────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 CNN编码器

```python
# 文件: TransModels/LDMMorph.py 第51-62行

def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
        bias=False, batchnorm=False):
    """
    卷积编码器块
    作用：将LDM特征的通道数调整到与Swin特征对齐
    """
    if batchnorm:
        # 使用BatchNorm的版本
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                     stride=stride, padding=padding, bias=bias),
            # Conv2d: 通道变换 in_channels → out_channels
            # kernel_size=3, padding=1: 保持空间尺寸不变

            nn.BatchNorm2d(out_channels),
            # BatchNorm: 加速训练，稳定梯度
            # 作用：归一化特征分布，加速收敛

            nn.PReLU())
            # PReLU: 带参数的ReLU
            # f(x) = max(0,x) + a * min(0,x)，a可学习
            # 相比ReLU，允许负值输出
    else:
        # 使用InstanceNorm的版本（默认）
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                     stride=stride, padding=padding, bias=bias),
            # 注意：这里没有bias，因为后续的InstanceNorm会抵消

            nn.PReLU())
            # 直接用PReLU激活

    return layer
```

### 3.3 解码器模块

```python
# 文件: TransModels/Decoder.py 第7-53行

class DecoderBlock(nn.Module):
    """
    解码器块
    作用：上采样 + 多尺度特征融合
    特点：三路skip connection（当前特征 + 两个skip特征）
    """
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()

        # 第一层：处理融合后的特征
        # 输入通道 = 当前特征 + skip1 + skip2
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels + skip_channels,
            out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )

        # 第二层：进一步处理
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )

        # 第三层：如果只有单路skip时的处理
        self.conv3 = Conv2dReLU(
            in_channels + skip_channels,  # 少一个skip
            out_channels,
            kernel_size=3, padding=1, use_batchnorm=use_batchnorm,
        )

        # 上采样：双线性插值
        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        # scale_factor=2: 将空间尺寸放大2倍
        # bicubic: 更好的插值质量，但计算量更大

    def forward(self, x, skip=None, skip2=None):
        """
        x: 当前层的特征 [B, C_in, H, W]
        skip: 来自浅层的特征 [B, C_skip, H/2, W/2]
        skip2: 来自另一个浅层的特征 [B, C_skip, H/2, W/2]
        """
        # 上采样到skip的分辨率
        x = self.up(x)
        # [B, C_in, H, W] → [B, C_in, 2H, 2W]

        # 处理分辨率不匹配
        if skip is not None:
            if skip.shape[2] != x.shape[2]:
                # 如果尺寸不匹配，裁剪x
                x = x[:,:,:skip.shape[2],:]

        # 第一路skip连接融合
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            # [B, C_in, H, W] + [B, C_skip, H, W] → [B, C_in+C_skip, H, W]

        # 第二路skip连接融合
        if skip2 is not None:
            if skip2.shape[2] != x.shape[2]:
                x = x[:,:,:skip2.shape[2],:]
            x = torch.cat([x, skip2], dim=1)
            # [B, C_in+C_skip, H, W] + [B, C_skip, H, W] → [B, C_in+2*C_skip, H, W]

            # 使用conv1处理（三路融合）
            x = self.conv1(x)
        if skip2 is None:
            # 只有单路skip时使用conv3（两路融合）
            x = self.conv3(x)

        # 最终卷积
        x = self.conv2(x)

        return x
```

### 3.4 形变场生成头

```python
# 文件: TransModels/Decoder.py 第103-109行

class RegistrationHead(nn.Sequential):
    """
    形变场生成头
    输出2通道向量场，表示每个像素的位移
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        # 卷积层：通道从in_channels变为2（x和y方向的位移）
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # Softsign激活函数
        # f(x) = x / (1 + |x|)
        # 取值范围：(-1, 1)
        # 作用：限制输出范围，防止形变过大
        sofsign = nn.Softsign()

        # 权重初始化为极小值
        # 作用：使网络从接近零的形变开始训练，稳定训练过程
        conv2d.weight = nn.Parameter(
            Normal(0, 1e-5).sample(conv2d.weight.shape)
        )
        # 偏置初始化为零
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))

        super().__init__(conv2d, sofsign)
```

---

## 四、空间变换层(STN)

### 4.1 原理说明

Spatial Transform Network (STN) 用于根据预测的形变场对图像进行变形。

### 4.2 代码逐行详解

```python
# 文件: utils/utils.py 第42-60行

class SpatialTransform(nn.Module):
    """
    空间变换模块
    作用：根据形变场对图像进行可微分的变形
    核心：grid_sample函数实现可微分的图像采样
    """
    def __init__(self):
        super(SpatialTransform, self).__init__()

    def forward(self, mov_image, flow, mod='bilinear'):
        """
        mov_image: 浮动图像 [B, C, H, W]
        flow: 形变场 [B, 2, H, W] - 第0通道是x位移，第1通道是y位移
        mod: 插值模式 'bilinear' 或 'nearest'
        返回: (采样网格, 变形后的图像)
        """
        # 获取图像尺寸
        h2, w2 = mov_image.shape[-2:]

        # 生成归一化坐标网格
        # 范围 [-1, 1]，与grid_sample的要求一致
        grid_h, grid_w = torch.meshgrid([
            torch.linspace(-1, 1, h2),  # 高度方向坐标
            torch.linspace(-1, 1, w2)    # 宽度方向坐标
        ])

        # 移到正确设备
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()

        # 锁定为非可训练参数
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)

        # 分离位移分量
        flow_h = flow[:,:,:,0]  # [B, H, W] 高度方向位移
        flow_w = flow[:,:,:,1]  # [B, H, W] 宽度方向位移

        # 计算变形后的坐标
        # 原始坐标 + 位移 = 变形后的采样位置
        disp_h = (grid_h + flow_h).squeeze(1)  # [H, W] → [B, H, W]
        disp_w = (grid_w + flow_w).squeeze(1)

        # 组合为采样网格
        # 形状: [B, H, W, 2] - 每个像素的(x,y)采样坐标
        sample_grid = torch.stack((disp_w, disp_h), 3)
        # 第0维是x(宽度)，第1维是y(高度) - grid_sample的要求

        # 可微分采样
        warped = torch.nn.functional.grid_sample(
            mov_image,           # 输入图像 [B, C, H, W]
            sample_grid,         # 采样坐标 [B, H, W, 2]
            mode=mod,            # 插值模式
            align_corners=True,  # 对齐角点
            padding_mode="border"  # 边界处理
        )

        return sample_grid, warped
```

**grid_sample工作原理：**
```
输入图像:          形变场:           采样网格:
┌────────────┐     ┌────────────┐     ┌────────────┐
│            │     │  每个像素   │     │  指向新    │
│   原始     │ →   │  的位移量   │ =   │  位置      │
│   图像     │     │  (dx, dy)  │     │  (x+dx,   │
│            │     │            │     │   y+dy)   │
└────────────┘     └────────────┘     └────────────┘

grid_sample根据采样网格在输入图像中采样：

对于位置(x,y)处的像素:
1. 找到采样点(x+dx, y+dy)
2. 使用双线性插值计算像素值:
   I'(x,y) = Σ I(i,j) * w(i,j)
   其中w是插值权重

可微分的原理：
- grid_sample内部使用grid索引
- 索引操作在反向传播时是可微的
- 因此整个变换过程可以梯度反传
```

---

## 五、训练流程总结

### 5.1 完整训练代码流程

```python
# 文件: train.py 第176-209行 (简化版)

# 外层循环：迭代次数
while step <= iteration:
    # 内层循环：数据批次
    for X, Y, segx, segy, _ in train_loader:

        # ===== LDM特征提取 =====
        mov_z = ldm_model.get_first_stage_encoding(
            ldm_model.encode_first_stage(X)
        ).detach()
        fix_z = ldm_model.get_first_stage_encoding(
            ldm_model.encode_first_stage(Y)
        ).detach()

        # 加噪过程
        noise = torch.randn_like(mov_z)
        x_noisy = ldm_model.q_sample(mov_z, t=1, noise=noise)
        y_noisy = ldm_model.q_sample(fix_z, t=1, noise=noise)

        # UNet去噪 + 特征提取
        outx = ldm_model.apply_model(x_noisy, t=1, cond=None, return_ids=True)
        outy = ldm_model.apply_model(y_noisy, t=1, cond=None, return_ids=True)

        # 特征拼接
        score0 = torch.cat([...], dim=1)
        score1 = torch.cat([...], dim=1)
        score2 = torch.cat([...], dim=1)
        score3 = torch.cat([...], dim=1)

        # ===== 配准网络前向 =====
        D_f_xy = model(X, Y, score0, score1, score2, score3)

        # ===== 图像变形 =====
        _, X_Y = transform(X, D_f_xy.permute(0, 2, 3, 1))

        # ===== 重新编码变形后图像 =====
        mov_z = ldm_model.get_first_stage_encoding(
            ldm_model.encode_first_stage(X_Y)
        ).detach()

        # ===== 损失计算 =====
        loss1 = beta * MSE(Y, X_Y) + (1-beta) * MSE(mov_z, fix_z)
        loss2 = smoothloss(D_f_xy)
        loss = loss1 + smooth * loss2

        # ===== 反向更新 =====
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 5.2 超参数说明

| 参数 | 默认值 | 作用 |
|-----|-------|------|
| lr | 1e-4 | 学习率 |
| bs | 1 | 批次大小 |
| iteration | 24001 | 训练迭代总数 |
| smooth | 0.01 | 平滑正则化权重 |
| beta | 0.6 | 像素/潜在空间损失权重 |
| t_enc | 1 | 扩散时间步 |

---

## 六、创新点总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LDM-Morph 创新点总结                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  创新1: LDM引导的语义特征提取                                         │
│  ├─ 原理：利用预训练扩散模型的语义理解能力                             │
│  ├─ 实现：LDM编码器 + UNet多尺度特征                                  │
│  └─ 作用：为配准提供高层解剖结构对应                                   │
│                                                                      │
│  创新2: LWSA局部窗口自注意力                                          │
│  ├─ 原理：Swin-Transformer的移位窗口机制                              │
│  ├─ 实现：窗口划分 + 循环移位 + SW-MSA                                │
│  └─ 作用：高效多尺度特征 + 全局感受野                                 │
│                                                                      │
│  创新3: LWCA局部全局交叉注意力                                        │
│  ├─ 原理：Query来自Swin，Key/Value来自LDM                            │
│  ├─ 实现：跨域注意力计算                                              │
│  └─ 作用：融合局部结构与语义信息                                      │
│                                                                      │
│  创新4: 分层度量损失函数                                              │
│  ├─ 像素空间MSE：细节对齐                                             │
│  ├─ 潜在空间MSE：语义匹配                                             │
│  └─ 平滑正则化：防止形变场折叠                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 参考文献

1. Wu, J., & Gong, K. (2024). LDM-Morph: Latent diffusion model guided deformable image registration. arXiv preprint arXiv:2411.15426.

2. Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. CVPR.

3. Liu, Z., et al. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. ICCV.

4. Chen, J., et al. (2022). TransMorph: Transformer for unsupervised medical image registration. MIDL.

---

*汇报人: [姓名]*
*日期: 2026年4月*
