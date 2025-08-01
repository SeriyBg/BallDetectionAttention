import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.channel_att(x)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa


class CSSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Squeeze and Excitation
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        # Spatial Squeeze and Excitation
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        chn_att = self.channel_att(x)
        # Spatial Attention
        spa_att = self.spatial_att(x)
        # Concurrent application (element-wise multiplication and sum)
        out = x * chn_att + x * spa_att
        return out


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # shape: (B, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # shape: (B, C, 1, W)

        reduced_channels = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.ReLU()

        self.conv_h = nn.Conv2d(reduced_channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(reduced_channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pooling along height and width separately
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, 1, W)

        # Concatenate along spatial dimension
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)

        # Shared transform
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split and apply attention
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class ECABlock(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # 1D conv along channel axis
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


class ApplyAttentionToList(nn.Module):
    def __init__(self, attention_type, channels_list):
        super().__init__()
        self.blocks = nn.ModuleList([
            attention_block(attention_type, ch) for ch in channels_list
        ])

    def forward(self, xs):
        return [att(x) for att, x in zip(self.blocks, xs)]


def attention_block(type, channels, reduction=16, k_size=3):
    assert type in ['se', 'cbam', 'csse', 'ca', 'eca']
    if type=='se':
        return SEBlock(channels, reduction)
    elif type=='cbam':
        return CBAM(channels, reduction)
    elif type=='csse':
        return CSSEBlock(channels, reduction)
    elif type=='ca':
        return CoordinateAttention(channels, reduction)
    elif type=='eca':
        return ECABlock(channels, k_size)

