import torch
import torch.nn as nn
from .basic import Conv


class SPPF(nn.Module):
    def __init__(self,in_dim,out_dim,expand_ratio=0.5,pooling_size=5,act_type="lrelu",norm_type="BN"):
        super(SPPF, self).__init__()
        inter_dim=int(in_dim * expand_ratio)
        self.out_dim=out_dim
        self.cv1=Conv(in_dim,inter_dim,k=1,act_type=act_type,norm_type=norm_type)
        self.cv2=Conv(inter_dim*4,out_dim,k=1,act_type=act_type,norm_type=norm_type)
        self.m=nn.MaxPool2d(kernel_size=pooling_size,stride=1,padding=pooling_size//2)


    def forward(self,x):
        x=self.cv1(x)
        y1=self.m(x)
        y2=self.m(y1)

        return self.cv2(torch.cat(x,y1,y2,self.m(y2),1))


def build_neck(cfg,in_dim,out_dim):
    """
    构建Neck模块
    cfg: 配置字典，包含Neck的参数
    in_dim: 输入通道数
    out_dim: 输出通道数
    """
    model=cfg['neck']
    print('Neck: {}'.format(model))

    if model == 'sppf':
        neck=SPPF(
            in_dim=in_dim,
            out_dim=out_dim,
            expand_ratio=cfg.get('expand_ratio', 0.5),
            pooling_size=cfg.get('pooling_size', 5),
            act_type=cfg.get('act_type', 'lrelu'),
            norm_type=cfg.get('norm_type', 'BN')
        )

    return neck
        