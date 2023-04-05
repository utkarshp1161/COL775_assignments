import torch
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features), requires_grad=False)

    def forward(self, x):
        if self.training:
            # Batch mean and variance
            batch_mean = x.mean(dim=0, keepdim=True)
            batch_var = x.var(dim=0, unbiased=False, keepdim=True)

            # Update running mean and variance
            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(self.momentum * batch_mean.data)
            self.running_var.mul_(1 - self.momentum)
            self.running_var.add_(self.momentum * batch_var.data)

            # Normalize input
            x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Normalize input using running mean and variance
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift output
        x = x * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return x

    
class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(InstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))
        self.running_mean = nn.Parameter(torch.zeros(num_features, 1, 1), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(num_features, 1, 1), requires_grad=False)

    def forward(self, x):
        N, C, H, W = x.size()
        mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        var = x.view(N, C, -1).var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / (var + self.eps).sqrt()
        x = x * self.weight + self.bias
        return x



class BatchInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=False)
        self.inn = nn.InstanceNorm2d(num_features, eps=eps, affine=False)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # BatchNorm
        bn_out = self.bn(x)
        
        # InstanceNorm
        in_out = self.inn(x)
        
        # Batch-Instance Norm
        b, c, _, _ = x.size()
        weight = self.weight.view(1, c, 1, 1).expand(b, c, 1, 1)
        bias = self.bias.view(1, c, 1, 1).expand(b, c, 1, 1)
        bin_out = weight * bn_out + (1 - weight) * in_out + bias
        
        return bin_out
    

class LayerNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = x.transpose(-1,-2) # x shape: (batch_size, num_features, sequence_length)
        mean = x.mean(dim=-1, keepdim=True) # shape: (batch_size, num_features, 1)
        std = x.std(dim=-1, keepdim=True) # shape: (batch_size, num_features, 1)
        x = (x - mean) / (std + self.eps)
        x = x.transpose(-1,-2) # shape: (batch_size, sequence_length, num_features)
        if self.affine:
            x = x * self.gamma + self.beta
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        assert C % self.num_groups == 0, 'Number of channels must be divisible by number of groups'
        x = x.view(N, self.num_groups, C // self.num_groups, H, W)
        mean = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        var = torch.var(x, dim=(2, 3, 4), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
