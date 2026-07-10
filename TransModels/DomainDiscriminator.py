"""
Domain Discriminator with Gradient Reversal Layer (GRL).

Architecture:
    Feature [B, 32, H, W]
        -> Conv(32, 64, 3,1,1) + LeakyReLU
        -> Conv(64, 128, 3,1,1) + LeakyReLU
        -> Conv(128, 64, 3,1,1) + LeakyReLU
        -> GlobalAvgPool -> FC(64, 1)
    Output: [B, 1] domain logit (scalar per sample)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    GRL: forward = identity, backward = -lambda * grad.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    Standalone GRL operator. Use ONLY on the path from feature to reg_model's
    parameters (i.e. place it between LDMMorph features and the discriminator
    forward call when computing the reg-side loss). Do NOT route the
    discriminator's own classification path through it.

    Standard DANN: two separate backward passes
        1. disc_loss = BCE(D(feat), label)        # discriminator params: normal grad
        2. reg_adv  = BCE(D(GRL(feat)), label)   # reg params:      reversed grad
    """

    def __init__(self):
        super().__init__()
        self._lambda = 0.0

    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, value):
        self._lambda = value

    def forward(self, x):
        return GradientReversalFunction.apply(x, self._lambda)


class DomainDiscriminator(nn.Module):
    """
    Patch-style discriminator on [B, 32, H, W] features.
    Produces a scalar domain logit per sample after global average pooling.

    IMPORTANT: This module is called TWICE per training step:
        1. With raw features     -> for discriminator update (normal gradient)
        2. With GRL(features)    -> for reg update       (reversed gradient)
    """
    def __init__(self, in_channels=32):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        return self.fc(feat)


class DomainAdversarialModule(nn.Module):
    """
    Standard DANN module.

    Two separate paths:
        - `disc_forward(feat)`: classifier only, normal grad -> updates discriminator
        - `reg_forward(feat)`:  GRL + classifier, reversed grad -> updates reg network

    Do NOT use a single forward(); that would reverse the discriminator's
    parameter gradients and cause both nets to "lie down" together.
    """
    def __init__(self, in_channels=32):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.discriminator = DomainDiscriminator(in_channels=in_channels)

    def set_lambda(self, lambda_):
        self.grl.lambda_ = lambda_

    def disc_forward(self, x):
        """Forward for discriminator update (no GRL)."""
        return self.discriminator(x)

    def reg_forward(self, x):
        """Forward for reg update (GRL applied -> reversed grad)."""
        x = self.grl(x)
        return self.discriminator(x)
