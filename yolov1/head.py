import torch
import torch.nn as nn

from .basic import Conv

class DecoupleHead(nn.Module):
    def __init__(self,cfg,in_dim,out_dim,num_classes=80):
        super().__init__()

        self.in_dim=in_dim
        self.num_cls_head=cfg['num_cls_head']
        self.num_reg_head=cfg['num_reg_head']
        self.act_type=cfg['head_act']
        self.norm_type=cfg['head_norm']

        cls_feats=[]
        self.cls_out_dim=max(out_dim,num_classes)
        
        for i in range(self.num_cls_head):
            if i==0:
                cls_feats.append(
                    Conv(in_dim,self.cls_out_dim,k=3,p=1,s=1,
                         act_type=self.act_type,
                         norm_type=self.norm_type,
                         depthwise=cfg['head_depthwise']
                         )
                )
            else:
                cls_feats.append(
                    Conv(self.cls_out_dim,self.cls_out_dim,k=3,p=1,s=1,
                         act_type=self.act_type,
                         norm_type=self.norm_type,
                         depthwise=cfg['head_depthwise']
                         )
                )

        reg_feats=[]
        self.reg_out_dim=max(out_dim,64)
        for i in range(self.num_reg_head):
            if i==0:
                reg_feats.append(
                    Conv(in_dim,self.reg_out_dim,k=3,p=1,s=1,
                         act_type=self.act_type,
                         norm_type=self.norm_type,
                         depthwise=cfg['head_depthwise']
                         )
                )
            else:
                reg_feats.append(
                    Conv(self.reg_out_dim,self.reg_out_dim,k=3,p=1,s=1,
                         act_type=self.act_type,
                         norm_type=self.norm_type,
                         depthwise=cfg['head_depthwise']
                         )
                )

        self.cls_feats=nn.Sequential(*cls_feats)
        self.reg_feats=nn.Sequential(*reg_feats)

    def forward(self,x):
        """
        前向传播
        x: 输入特征图，形状为(batch_size, in_dim, height, width)
        """
        cls_out=self.cls_feats(x)
        reg_out=self.reg_feats(x)

        return cls_out, reg_out


def build_head(cfg,in_dim,out_dim,num_classes=80):
    """
    构建Head模块
    cfg: 配置字典，包含Head的参数
    in_dim: 输入通道数
    out_dim: 输出通道数
    num_classes: 类别数量，默认为80
    """
    head=DecoupleHead(cfg,in_dim,out_dim,num_classes)

    return head