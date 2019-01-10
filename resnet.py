import torch.nn as nn
import torch.nn.functional as F


#Conv2d with padding=same effect
#only support square kernels and stride=1, dilation=1, groups=1
class Conv2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):

    def __init__(self, num_in, num_out, down_sample_stride):
        super(ResBlock, self).__init__()

        self.conv1 = Conv2dSame(num_in, num_in, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(num_in)

        if(down_sample_stride==2):
            self.conv2 = nn.Conv2d(num_in, num_out, kernel_size=3, stride=2, padding=1)
            self.identity_map = nn.Conv2d(num_in, num_out, kernel_size=1, stride=2)
        else:
            self.conv2 = Conv2dSame(num_in, num_out, kernel_size=3)
            self.identity_map = nn.Conv2d(num_in, num_out, kernel_size=1, stride=1)

        self.conv2_bn = nn.BatchNorm2d(num_out)

    def forward(self, x):

        x = F.relu(self.conv1_bn(self.conv1(x)))
        identity = self.identity_map(x)
        x = F.relu(self.conv2_bn(self.conv2(x) + identity)) # identity shortcut ###############

        return x

class ResNet(nn.Module):

    def __init__(self, num_block, num_class=10):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.sub_module16 = self.subModule(num_block, 16, 32)
        self.sub_module32 = self.subModule(num_block, 32, 64)
        self.sub_module64 = self.subModule(num_block, 64, 64)

        self.fc1 = nn.Linear(64, num_class) # num of filters: 64


    def subModule(self, num_block, num_in, num_out):

        block_container = []
        for i in range(num_block-1):
            block_container.append(ResBlock(num_in, num_in, 1))

        # Reduce feature map size & increase(double) filters
        block_container.append(ResBlock(num_in, num_out, 2))

        return nn.Sequential(*block_container)


    def forward(self, x):

        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.sub_module16(x) # feature map size: 32 -> 16
        x = self.sub_module32(x) # feature map size: 16 -> 8
        x = self.sub_module64(x) # feature map size: 8 -> 4 at the end.

        x = F.avg_pool2d(x, 4)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)

        return x




