import torch
import torch.nn as nn

import numpy as np

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head


class YOLOv1(nn.Module):
    def __init__(self,
                cfg,
                device,
                img_size=None,
                num_classes=20,
                conf_thresh=0.5,
                trainable=False,
                deploy=False,
                num_class_agnostic:bool=False,# nms下是否考虑预测框的类别信息
                ):
        super(YOLOv1, self).__init__() ## 继承父类

        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.trainable = trainable
        self.stride=32
        self.deploy = deploy
        self.num_class_agnostic = num_class_agnostic

        # backbone net
        self.backbone,feat_dim=build_backbone(
            cfg['backbone'],
            trainable&cfg['pretrained']
        )

        # neck net
        self.neck=build_neck(
            cfg,feat_dim,out_dim=512
        )
        head_dim=self.neck.out_dim

        # hean net
        self.head=build_head(
            cfg,
            head_dim,head_dim,num_classes
        )


        # 预测层
        self.obj_pred=nn.Conv2d(head_dim,1,kernel_size=1)
        self.cls_pred=nn.Conv2d(head_dim,num_classes,kernel_size=1)
        self.reg_pred=nn.Conv2d(head_dim,4,kernel_size=1)

    def create_grid(self,fmp_size):
        """
        生成G矩阵，其中每个元素表示对应位置的网格坐标
        """
        ws,hs=fmp_size

        gy,gx=torch.meshgrid([torch.arrange(hs),torch.arange(ws)])

        # [H,W,2]
        g_xy=torch.stack([gx,gy],dim=-1).float() # [H,W,2]

        # [H,W,2] -> [HW,2]
        g_xy=g_xy.view(-1,2).to(self.device) # [H*W,2]

        return g_xy


    def decode_boxes(self,pred,fmp_size):
        pass


        
    @torch.no_grad()
    def inference(self,x):
        # backbone
        feat=self.backbone(x)

        # neck
        feat=self.neck(feat)

        # head
        cls_feat,reg_feat=self.head(feat)

        # 预测层
        obj_pred=self.obj_pred(reg_feat)
        cls_pred=self.cls_pred(cls_feat)
        reg_pred=self.reg_pred(reg_feat)
        fmp_size=obj_pred.shape[-2:]  # [H,W]

        # view : [B,C,H,W]-> [B,H,W,C] -> [B,H*W,C]
        obj_pred=obj_pred.permute(0,2,3,1).contiguous().flatten(1,2)
        cls_pred=cls_pred.permute(0,2,3,1).contiguous().flatten(1,2)
        reg_pred=reg_pred.permute(0,2,3,1).contiguous().flatten(1,2)

        # batch default is 1
        obj_pred=obj_pred[0]
        cls_pred=cls_pred[0]
        reg_pred=reg_pred[0]

        # 每个框的得分
        scores=torch.sqrt(obj_pred.sigmoid()*cls_pred.sigmoid())

        # 结算边界框的预测结果，皈依化框的坐标:[H*W,4]
        bboxes=self.decode_boxes(reg_pred,fmp_size)

        if self.deploy:
            # [n_anchors_all,4+C]
            outputs=torch.cat([bboxes,scores],dim=-1)
        else:
            scores=scores.cpu().numpy()
            bboxes=bboxes.cpu().numpy()

            # 后处理
            bboxes,scores,labels=self.post_process(bboxes,scores)

            outputs={
                "scores": scores,
                "labels": labels,
                "bboxes": bboxes
            }

        return outputs