import torch.nn as nn
from mmseg.models.decode_heads import FCNHead
from mmseg.ops import resize
from mmcv.cnn import ConvModule


class NestedFCNHead(nn.Module):
    def __init__(self, origin_shape=[128,128]):
        super().__init__()
        # self.head = FCNHead(in_channels=512,
        #        in_index=0, 
        #        channels=256, 
        #        num_convs=1,
        #        concat_input=False,
        #        dropout_ratio=0.1,
        #        num_classes=2,
        #        norm_cfg=dict(type='BN', requires_grad=True),
        #        loss_decode=None,
        #        align_corners=False
        #        )

        kernel_size = 3
        dilation=1

        self.in_channels=512
        self.in_index=0
        self.channels=256
        num_convs=1
        concat_input=False
        dropout_ratio=0.1
        self.num_classes=2
        self.norm_cfg=dict(type='BN', requires_grad=False)
        self.act_cfg=dict(type="ReLU")
        self.conv_cfg=None


        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)


        self.origin_shape = origin_shape

    def forward(self, inputs):
        # inputs are list of NestedTesnor
        # torch.Size([256, 512, 16, 16]) # 1/8, 1/8
        # torch.Size([256, 1024, 8, 8])
        # torch.Size([256, 2048, 4, 4])

        srcs = []
        for l, feat in enumerate(inputs):
            src, _ = feat.decompose()
            srcs.append(src)

        # preds = self.head(srcs) # output: 1, 2, 16, 16
        x = srcs[self.in_index]
        x = self.convs(x)
        preds = self.conv_seg(x)

        preds = resize(
                        preds,
                        size=self.origin_shape,
                        mode='bilinear',
                        align_corners=False,
                        warning=False)
        
        return preds