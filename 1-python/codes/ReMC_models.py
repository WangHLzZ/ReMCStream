from __future__ import print_function
# import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU

# dim>=256
class EncoderVAE1(nn.Module):
    def __init__(self, num_hidden_units=2, s=2, num_classes=10, feature_dim=10):
        super(EncoderVAE1, self).__init__()
        self.scale = s
        self.dce = dce_loss(num_classes, num_hidden_units)
        self.encoderFC = nn.Sequential(
            nn.Linear(feature_dim, int(0.6*feature_dim)),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.Linear(int(0.7*feature_dim), int(0.4*feature_dim)),
            # nn.ReLU(),
            nn.Linear(int(0.6*feature_dim), int(0.3*feature_dim)),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.Linear(int(0.5*feature_dim), int(0.3*feature_dim)),
            # nn.ReLU(),
        )
        self.mean = nn.Linear(int(0.3*feature_dim), num_hidden_units)
        self.logvar = nn.Linear(int(0.3*feature_dim), num_hidden_units)

    def sampler(self, mean, logvar):
        std = logvar.mul_(0.5).exp_()
        # eps = torch.FloatTensor(std.size()).normal_().cuda()
        eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul_(std).add_(mean)

    def forward(self, x):
        hidden = self.encoderFC(x)
        # hidden = hidden.view(len(x), -1)
        mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)  # output feature
        # centers, dist = self.dce(latent_z)
        # output = F.log_softmax(self.scale * dist, dim=1)
        centers, dist = 0,1
        output = 0
        return mean, logvar, latent_z, centers, dist, output

# dim<=32
class EncoderVAE2(nn.Module):
    def __init__(self, num_hidden_units=2, s=2, num_classes=10, feature_dim=10):
        super(EncoderVAE2, self).__init__()
        self.scale = s
        self.dce = dce_loss(num_classes, num_hidden_units)
        
        self.encoderFC1 = nn.Sequential(
            nn.Linear(feature_dim, int(4*feature_dim)),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(int(4*feature_dim), int(2*feature_dim)),
            nn.ReLU(),
            # nn.Tanh(),
        )
        self.mean1 = nn.Linear(int(2*feature_dim), num_hidden_units)
        self.logvar1 = nn.Linear(int(2*feature_dim), num_hidden_units)

    def sampler(self, mean, logvar):
        std = logvar.mul_(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        # eps = torch.FloatTensor(std.size()).normal_().cuda()
        return eps.mul_(std).add_(mean)

    def forward(self, x):
        hidden = self.encoderFC1(x)
        # hidden = hidden.view(len(x), -1)
        mean, logvar = self.mean1(hidden), self.logvar1(hidden)
        latent_z = self.sampler(mean, logvar)  # output feature
        # centers, dist = self.dce(latent_z)
        # output = F.log_softmax(self.scale * dist, dim=1)
        centers, dist = 0,1
        output = 0
        return mean, logvar, latent_z, centers, dist, output

# dim:32~256
class EncoderVAE3(nn.Module):
    def __init__(self, num_hidden_units=2, s=2, num_classes=10, feature_dim=10):
        super(EncoderVAE3, self).__init__()
        self.scale = s
        self.dce = dce_loss(num_classes, num_hidden_units)

        self.encoderFC2 = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Tanh(),
        )
        self.mean2 = nn.Linear(64, num_hidden_units)
        self.logvar2 = nn.Linear(64, num_hidden_units)

    def sampler(self, mean, logvar):
        std = logvar.mul_(0.5).exp_()
        # eps = torch.FloatTensor(std.size()).normal_().cuda()
        eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul_(std).add_(mean)

    def forward(self, x):
        hidden = self.encoderFC2(x)
        # hidden = hidden.view(len(x), -1)
        mean, logvar = self.mean2(hidden), self.logvar2(hidden)
        latent_z = self.sampler(mean, logvar)  # output feature
        # centers, dist = self.dce(latent_z)
        # output = F.log_softmax(self.scale * dist, dim=1)
        centers, dist = 0,1
        output = 0
        return mean, logvar, latent_z, centers, dist, output
    

class DecoderVAE1(nn.Module):
    def __init__(self, num_hidden_units=2, feature_dim=10, num_classes=10):
        super(DecoderVAE1, self).__init__()
        # dim>=256
        self.decoderFC = nn.Sequential(
            nn.Linear(int(0.3*feature_dim), int(0.6*feature_dim)),
            nn.ReLU(),
            # nn.Tanh(),
            # nn.Linear(int(0.1*feature_dim), int(0.4*feature_dim)),
            # nn.ReLU(),
            # nn.Linear(int(0.5*feature_dim), int(0.8*feature_dim)),
            # nn.ReLU(),
            nn.Linear(int(0.6*feature_dim), feature_dim),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Sigmoid()
        )
        self.decoder_dense = nn.Sequential(
            nn.Linear(num_hidden_units, int(0.3*feature_dim)),
            nn.ReLU(),
            # nn.Tanh(),
        )
        
    def forward(self, x):
        x1 = self.decoder_dense(x)
        output = self.decoderFC(x1)
        return output

class DecoderVAE2(nn.Module):
    def __init__(self, num_hidden_units=2, feature_dim=10, num_classes=10):
        super(DecoderVAE2, self).__init__()
        
        # dim<=32
        self.decoderFC1 = nn.Sequential(
            nn.Linear(int(2*feature_dim), int(4*feature_dim)),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(int(4*feature_dim), feature_dim),
            # nn.ReLU(),
            nn.Sigmoid()
        )
        self.decoder_dense1 = nn.Sequential(
            nn.Linear(num_hidden_units, int(2*feature_dim)),
            nn.ReLU(),
            # nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.decoder_dense1(x)
        output = self.decoderFC1(x1)
        return output

class DecoderVAE3(nn.Module):
    def __init__(self, num_hidden_units=2, feature_dim=10, num_classes=10):
        super(DecoderVAE3, self).__init__()
        # dim:32~256
        self.decoderFC2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(128, feature_dim),
            # nn.ReLU(),
            nn.Sigmoid()
        )
        self.decoder_dense2 = nn.Sequential(
            nn.Linear(num_hidden_units, 64),
            nn.ReLU(),
            # nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.decoder_dense2(x)
        output = self.decoderFC2(x1)
        return output

class ConfidenceVae(nn.Module):
    def __init__(self, num_hidden_units=2, num_classes=2):
        super(ConfidenceVae, self).__init__()

        self.confidence1 = nn.Sequential(
            nn.Linear(num_hidden_units, int(2*num_hidden_units)),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(int(2*num_hidden_units), int(4*num_hidden_units)),
            nn.ReLU(),
            # nn.Tanh(),
        )
        self.fc1 = nn.Linear(int(4*num_hidden_units), num_classes)
        self.con1 = nn.Linear(int(4*num_hidden_units), 1)

        self.confidence = nn.Sequential(
            nn.Linear(num_hidden_units, 128),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Tanh(),
        )
        self.fc = nn.Linear(64, num_classes)
        self.con = nn.Linear(64, 1)

    def forward(self, x):
        output = self.confidence1(x)
        fc = self.fc1(output)
        conf = self.con1(output)
        # output = self.confidence(x)
        # fc = self.fc(output)
        # conf = self.con(output)

        return fc, conf

class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim, init_weight=True):

        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        # self.centers = nn.Parameter(
        #     torch.randn(self.feat_dim, self.n_classes).cuda(),
        #     requires_grad=True) 
        self.centers = nn.Parameter(
            torch.randn(self.feat_dim, self.n_classes),
            requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):

        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, -dist


def regularization(features, centers, labels):
    distance = (features - torch.t(centers)[labels])

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance
