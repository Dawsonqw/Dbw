import torch
import torch.nn as nn

class SiLU(nn.Module):
    """
    公式：
    SiLU(x) = x * sigmoid(x)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_conv2d(c1,c2,k,p,s,d,g,bias=False):
    """
    获取卷积层
    c1: 输入通道数
    c2: 输出通道数
    k: 卷积核大小
    p: 填充
    s: 步长
    d: 膨胀率
    g: 分组卷积的组数
    bias: 是否使用偏置
    """
    return nn.Conv2d(c1, c2, kernel_size=k, padding=p, stride=s, dilation=d, groups=g, bias=bias)


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return SiLU()

def get_norm(norm_type,dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32,num_channels=dim)


class Conv(nn.Module):
    def __ini__(self,c1,c2,k=1,p=0,s=1,d=1,act_type='lrelu',norm_type='BN',depthwise=False):
        """
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核大小
        p: 填充
        s: 步长
        d: 膨胀率
        act_type: 激活函数类型
        norm_type: 归一化类型
        depthwise: 是否使用深度卷积
        """
        super(Conv, self).__init__()
        add_bias=False if norm_type else True
        convs=[]
        if depthwise:
            convs.append(get_conv2d(c1, c1, k, p, s, d, c1, add_bias))
            # depthwise卷积
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
        else:
            convs.append(get_conv2d(c1, c2, k, p, s, d, 1, add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)
