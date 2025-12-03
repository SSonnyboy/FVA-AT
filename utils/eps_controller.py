import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class EPS_ruler():
    def __init__(self, labels, default_eps, max_eps, total_epochs, rew_epoch, device, T):
        self.set_lib = torch.zeros(labels.shape[0], dtype=torch.float).to(device)
        self.default_eps = default_eps
        self.max_eps = max_eps
        self.total_epochs = total_epochs
        self.device = device
        self.cur_max_eps = self.default_eps
        self.rew_epoch = rew_epoch
        self.T = T
        self.eps_increase_step = (max_eps - default_eps)/(total_epochs-rew_epoch)

    def update_sur_max_eps(self, epoch):
        if epoch>self.rew_epoch:
            self.cur_max_eps += self.eps_increase_step
        
    def update_lib(self, idx, feats_margin):
        self.set_lib[idx] = feats_margin

    def get_eps(self, idx):
        margin = self.set_lib[idx]  
        reweight_unnorm = torch.exp((margin - margin.max()))  # numerical stability
        eps_values = self.default_eps + reweight_unnorm * (self.cur_max_eps - self.default_eps)
        return eps_values
    