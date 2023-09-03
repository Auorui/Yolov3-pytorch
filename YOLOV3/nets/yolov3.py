from collections import OrderedDict
import torch
import torch.nn as nn
from yolo3.nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    """
    conv2d代替池化层
    :param filter_in: 输入通道数
    :param filter_out: 输出通道数
    :param kernel_size: 卷积核大小
    :return: 卷积块,包含卷积层、BN层和激活层
    """
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

def make_last_layers(filters_list, in_filters, out_filter):
    """
    FPN金字塔特征提取
    前5个卷积层是用于特征提取:
    conv1:减小通道数
    conv2:使用3x3卷积核提取特征
    conv3:再次减小通道数
    conv4:继续使用3x3卷积核提取特征
    conv5:进一步减小通道数
    这几层卷积的作用是聚集和提炼前面网络层输出的特征。
    后2个卷积层是用于获得最终的预测结果:
    conv6:使用3x3卷积核提取特征
    conv7:使用1x1卷积核进行分类预测,输出通道数等于类别数
    """
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),

        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#
        #   生成darknet53的主干模型,获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #   实际就是前向传播当中的 out3, out4, out5
        #---------------------------------------------------#
        self.backbone = darknet53(pretrained=pretrained)

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters

        #----------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言（类别20）,每个特征层预测3个边界框
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 3*(20+5) = 75
        #   对于coco数据集而言（类别80）
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 3*(80+5) = 255
        #----------------------------------------------------------------------------#
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)
        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,final_out_filter0(255),13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        #   对前五个卷积层进行特征提取,后面的层进行预测输出
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2