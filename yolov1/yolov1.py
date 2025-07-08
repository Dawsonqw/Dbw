import torch
import torch.nn as nn


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

        # neck net

        # hean net

        # 预测层
        