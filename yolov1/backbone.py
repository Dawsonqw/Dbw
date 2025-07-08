import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# Basic Module for ResNet
def conv3x3(in_planes,out_planes,stride=1):
    '''
    3x3卷积层，默认步长为1
    in_planes:输入通道数
    out_planes:输出通道数
    stride:卷积步长，默认为1
    '''
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)


def conv1x1(in_planes,out_planes,stride=1):
    '''
    1x1卷积层，默认步长为1
    in_planes:输入通道数
    out_planes:输出通道数
    stride:卷积步长，默认为1
    '''
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

class BasicBlock(nn.Module):
    expansion = 1 # 扩展因子，BasicBlock的输出通道数等于输入通道数的 expansion 倍
    def __init__(self,inplanes,planes,stride=1,dawnsample=None):
        '''
        残差块
        inplanes:输入通道数
        planes:输出通道数
        stride:卷积步长，默认为1
        dawnsample:下采样，默认为None
        '''
        super(BasicBlock,self).__init__()
        self.conv1=conv3x3(inplanes,planes,stride) # 第一个卷积层
        self.bn1=nn.BatchNorm2d(planes) # 批归一化层
        self.relu=nn.ReLU(inplace=True) # 激活函数
        self.conv2=conv3x3(planes,planes) # 第二个卷积
        self.bn2=nn.BatchNorm2d(planes) # 批归一化层
        self.downsample=dawnsample
        self.stride=stride

    def forward(self,x):
        '''
        前向传播
        x:输入数据
        '''
        identity=x # 残差连接的输入

        out=self.conv1(x) # 第一个卷积层
        out=self.bn1(out) # 批归一化层
        out=self.relu(out) # 激活函数

        out=self.conv2(out)
        out=self.bn2(out)

        if self.downsample is not None: # 如果有下采样层
            identity=self.downsample(x) # 对输入进行下采样
        out+=identity # 残差连接
        out=self.relu(out) # 激活函数

        return out

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        self.conv1=conv1x1(inplanes,planes)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=conv3x3(planes,planes,stride)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv3=conv1x1(planes,planes*self.expansion)
        self.bn3=nn.BatchNorm2d(planes*self.expansion)

        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        identity=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample is not None:
            identity=self.downsample(x)

        out+=identity
        out=self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self,block,layers,zero_init_residual=False):
        '''
        block:resnet分为BasicBlock和Bottleneck两种类型，BasicBlock适用于层数较小的网络，Bottleneck适用于层数多的网络
        layers:每个stage残差块的数量，通常是一个列表，例如[2, 2, 2, 2]表示每个阶段有2个block
        zero_init_residual:是否将残差连接的最后一层的权重初始化为0，默认为False
        '''
        super(ResNet,self).__init__()
        self.inplanes=64  # 输入通道数，初始为64

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True) # inplace=True表示在原地修改数据，节省内存
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # 池化层,用于下采样

        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)


    def _make_layer(self,block,planes,blocks,stride=1):
        '''
        block:残差块的类型
        planes:当前层块的输出通道数
        blocks:当前层块的残差块数量
        stride:卷积步长，默认为1
        '''
        # 判断是否需要下采样层
        downsample=None
        # 如果输入通道和残差块的输出通道不一致，或者步长不为1时需要下采样
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion, stride=stride),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers=[]

        layers.append(block(self.inplanes,planes,stride,downsample)) # 添加第一个残差块

        self.inplanes=planes*block.expansion # 更新输入通道数
        for _ in range(1,blocks): # 添加剩余的残差块
            layers.append(block(self.inplanes,planes))

        return nn.Sequential(*layers) # 返回一个顺序容器，包含所有的残差块

    def forward(self,x):
        '''
        前向传播
        x:输入数据 (Tensor)->[batch_size, channels, height, width]
        返回:
        x:输出数据 (Tensor)->[batch_size, channels, height/32, width/32]
        '''
       
        # conv: output_size=(input_size+2*Padding-kernel_size)/stride+1
        c1=self.conv1(x) # [B,C,H/2,W/2]  H+2*3-7/2+1=(H-1)/2+1
        c1=self.bn1(c1) # [B,C,H/2,W/2]
        c1=self.relu(c1) # [B,C,H/2,W/2]
        # maxpool: output_size=(input_size+2*Padding-kernel_size)/stride+1
        c2=self.maxpool(c1) # [B,C,H/4,W/4] 

        c2=self.layer1(c2) # [B,C,H/4,W/4]

        c3=self.layer2(c2) # [B,C,H/8,W/8]
        c4=self.layer3(c3) # [B,C,H/16,W/16]
        c5=self.layer4(c4) # [B,C,H/32,W/32]

        return c5

    
def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False) # state_dict作用：表示加载权重时允许部分参数不匹配（如 fc 层），本网络没有fc 层
    return model

# aother resnet todo...


def build_backbone(model_name="resnet18",pretrained=False):
    feat_dim=512 # 标识主干网络输出的特征维度，供后续网络使用
    model=None
    if model_name == "resnet18":
        model = resnet18(pretrained=pretrained)
        feat_dim = 512
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models: resnet18.")
    return model, feat_dim

if __name__ == "__main__":
    model,feat_dim = build_backbone(model_name="resnet18", pretrained=True)
    print(model)

    input_tensor = torch.randn(1, 3, 512, 512)  # Example input tensor
    output = model(input_tensor)