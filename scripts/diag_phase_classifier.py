"""
诊断脚本：MotionEncoder + phase_mlp 能否单独学相位分类
不走 VQGAN，不走 L_rec/L_swap，就测一个分类任务。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ldm.data.xcat_seq_grouped import (
    XCATSeqGroupedTrain,
    XCATSeqGroupedValidation,
    xcat_seq_grouped_collate_fn,
)


# ============================================================
# 模型：纯 MotionEncoder + phase_mlp（与 dpt_vqgan.py 完全一致）
# ============================================================
class MotionEncoder(nn.Module):
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
        h = self.backbone(x)           # (B, 128, 32, 32)
        h = self.pool(h).flatten(1)    # (B, 128)
        return self.proj(h)            # (B, m_dim)


class PhaseClassifier(pl.LightningModule):
    def __init__(self, m_dim=64, num_phases=9, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.motion_encoder = MotionEncoder(m_dim=m_dim)
        self.phase_mlp = nn.Sequential(
            nn.Linear(m_dim, 128),
            nn.SiLU(),
            nn.Linear(128, num_phases),
        )

    def forward(self, x):
        m = self.motion_encoder(x)
        return self.phase_mlp(m)

    def _shared_step(self, batch, split="train"):
        images = batch["images"]           # (B, 9, 1, H, W)
        phases = batch["phases"]            # (B, 9)
        B, num_frames = images.shape[:2]

        x_flat = images.reshape(-1, 1, images.shape[3], images.shape[4])  # (B*9, 1, H, W)
        phases_flat = phases.reshape(-1)                                    # (B*9,)

        logits = self(x_flat)                     # (B*9, num_phases)
        loss = F.cross_entropy(logits, phases_flat)
        acc = (logits.argmax(1) == phases_flat).float().mean()

        self.log(f"{split}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{split}/phase_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    train_ds = XCATSeqGroupedTrain()
    val_ds   = XCATSeqGroupedValidation()

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        num_workers=4, collate_fn=xcat_seq_grouped_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=4, collate_fn=xcat_seq_grouped_collate_fn, pin_memory=True,
    )

    model = PhaseClassifier(m_dim=64, num_phases=9, lr=1e-3)

    logger = pl.loggers.TensorBoardLogger("logs/", name="phase_classifier_diag")
    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val/phase_acc", mode="max", patience=10),
        pl.callbacks.ModelCheckpoint(monitor="val/phase_acc", mode="max", save_top_k=1),
    ]

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)
