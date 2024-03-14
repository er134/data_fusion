import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)      

        logpt = torch.log_softmax(input, 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, predict, target):
        pt = predict
        loss = - ((1 - self.alpha) * ((1 - pt + 1e-5) ** self.gamma) * (target * torch.log(pt + 1e-5)) + self.alpha * (
                (pt + 1e-5) ** self.gamma) * ((1 - target) * torch.log(1 - pt + 1e-5)))
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


class Combined_Bce_Dice_Loss(nn.Module):
    def __init__(self,dice_weight=torch.tensor(1.0).cuda(),bce_weight=torch.tensor(0.).cuda()):
        super(Combined_Bce_Dice_Loss,self).__init__()
        self.dice_w = dice_weight
        self.bce_w = bce_weight
        self.diceloss = SoftDiceLoss(0.)
        self.bceloss = nn.BCELoss()#Balance_Loss(nepochs)
    
    def forward(self,pred, target, mask):
        p = pred * mask
        t = target * mask
        bce = self.bceloss(p, t)
        dice = self.diceloss(p, t)
        return self.bce_w*bce + self.dice_w*dice

class ModifiedOhemLoss(nn.Module): # only available for binary-classification

    def __init__(self, thresh=1.0, min_kept=256): # OHEM ≈ BCE when thresh = 1.0
        super(ModifiedOhemLoss, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = torch.nn.BCELoss(reduction='none')

    def forward(self, pred, target, valid_mask=None):
        pred = pred.view(-1)
        target = target.view(-1)
        if valid_mask is None:
            valid_mask = torch.ones_like(pred)
        else:
            valid_mask = valid_mask.view(-1)
        pos_mask = (target.ne(0) * valid_mask).bool()    # True for positive sample
        neg_mask = (target.eq(0) * valid_mask).bool()    # True for negative sample
        neg_prob = 1 - torch.sigmoid(pred)               # (1-p) for negative probs
        neg_prob = neg_prob.masked_fill_(~neg_mask, 1.1)

        num_valid = min(self.min_kept, neg_mask.sum())
        if num_valid > 0:
            index = neg_prob.argsort()  # probs from small to large
            thres_index = index[num_valid - 1]
            if neg_prob[thres_index] > self.thresh:
                threshold = neg_prob[thres_index]
            else:
                threshold = self.thresh
            kept_mask = neg_prob.le(threshold).long() + pos_mask.long()
        else:
            kept_mask = pos_mask.long()

        loss = self.criterion(pred, target) * kept_mask
        return loss
