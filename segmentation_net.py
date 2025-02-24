import torch
import torch.nn as nn
import torch.nn.functional as F
from primary_net import Embedding
from hypernetwork_modules import HyperNetwork

from resnet_blocks import ResNetBlock, IdentityLayer

class SegmentationNetwork(nn.Module):
    def __init__(self, num_classes, z_dim=64):
        super(SegmentationNetwork, self).__init__()
        # Initial Convolution
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.z_dim = z_dim
        self.hope = HyperNetwork(z_dim=self.z_dim)
        
        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                            [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                            [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]

        
        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs = nn.ModuleList()
        for i in range(36):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim))
            
        # # Upsampling layers
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)  # This produces the segmentation mask

        # # Upsampling layers' filter sizes and latent vectors' sizes
        # self.upsample_filter_size = [[32, 64], [16, 32], [num_classes, 16]]
        # self.upsample_zs_size = [[1, 1], [1, 1], [1, 1]]

        # # Embeddings for upsampling layers
        # self.upsample_zs = nn.ModuleList()
        # for i in range(3):
        #     self.upsample_zs.append(Embedding(self.upsample_zs_size[i], self.z_dim))


    def forward(self, x, shared_embeddings = None):
        # add the shared_embeddings
        if shared_embeddings is not None:
            self.zs = shared_embeddings

        x = F.relu(self.bn1(self.conv1(x)))
        
        for i in range(18):
            w1 = self.zs[2*i](self.hope)
            w2 = self.zs[2*i+1](self.hope)
            x = self.res_net[i](x, w1, w2)
            
        # # Upsampling steps
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x) # No activation here as you might want to apply softmax later

        #  # Upsampling steps with hypernetwork-generated weights
        # for i in range(3):
        #     w_up = self.upsample_zs[i](self.hope)
        #     x = F.conv_transpose2d(x, w_up, stride=2, padding=1, output_padding=1)
        #     if i < 2:  # Apply ReLU for the first two upsampling layers
        #         x = F.relu(x)
        # x = F.softmax(x, dim=1)

        
        return x
