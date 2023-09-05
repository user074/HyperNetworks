import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# Initialization:

# f_size: The spatial size of the filter (e.g., 3 for a 3x3 filter).
# z_dim: The dimension of the input vector z.
# out_size: The number of output channels of the kernel.
# in_size: The number of input channels of the kernel.
# Parameters:

# w1, b1: Parameters used to generate the convolutional kernel weights.
# w2, b2: Parameters used to transform the input z vector.
# Forward Method:

# The input z is linearly transformed using w2 and b2, resulting in h_in.
# This transformation essentially captures the relationship between the input z and the final kernel.
# It reshapes h_in to have a size of (self.in_size, self.z_dim).
# h_in is then transformed using w1 and b1 to produce h_final.
# This transformation generates the actual convolutional kernel.
# Finally, h_final is reshaped into a tensor with the shape of a convolutional kernel: (self.out_size, self.in_size, self.f_size, self.f_size). This is the tensor that will be used as a convolutional filter in the main/target network.
class HyperNetwork(nn.Module):

    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).cuda(),2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)).cuda(),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)).cuda(),2))

    def forward(self, z):

        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel





