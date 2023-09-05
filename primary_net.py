import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from hypernetwork_modules import HyperNetwork
from resnet_blocks import ResNetBlock


class Embedding(nn.Module):
# Initialization (__init__ method):

# z_num: A tuple containing two integers, denoting the dimensions in which the latent vectors will be organized. For example, (h, k) means there will be h groups, each containing k latent vectors.
# z_dim: An integer denoting the dimensionality of each latent vector.
# Forward method:

# hyper_net: The hypernetwork model that will take each latent vector and generate the corresponding weight tensor.


    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim

        h,k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))

#The output of the forward method of the Embedding class is a concatenated tensor of weight tensors. 
# This output tensor is created by feeding each latent vector in the z_list to the hyper_net. 
# The resulting weight tensors are concatenated along specified dimensions to produce the output.
    def forward(self, hyper_net):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)


class PrimaryNetwork(nn.Module):
# Initialization (__init__ method):

# z_dim: An integer denoting the dimensionality of the latent vector. This is used as an input dimension for the HyperNetwork within the PrimaryNetwork.
# Forward method:

# x: An input tensor, which represents the images you're processing. Based on the code, the expected shape for x is [batch_size, 3, 32, 32] which corresponds to a batch of 32x32 RGB images.
    def __init__(self, z_dim=64):
        super(PrimaryNetwork, self).__init__()
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

        self.global_avg = nn.AvgPool2d(64)
        self.final = nn.Linear(64,20)

    def forward(self, x, shared_embeddings = None):
        # add the shared_embeddings
        if shared_embeddings is not None:
            self.zs = shared_embeddings

    # The output of the forward method of the PrimaryNetwork class is the processed tensor x, which represents the predictions of the network for the given input images. The shape of this output tensor is [batch_size, 10], corresponding to the 10 class predictions for the CIFAR-10 dataset.

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(18):
            # if i != 15 and i != 17:
            w1 = self.zs[2*i](self.hope)
            w2 = self.zs[2*i+1](self.hope)
            x = self.res_net[i](x, w1, w2)
            # print("After ResNet Block", i, "Shape:", x.shape)

        x = self.global_avg(x)
        # print("After Global Avg Pool:", x.shape)
        x = self.final(x.view(-1,64))
        # print("After Final Layer:", x.shape)

        return x
