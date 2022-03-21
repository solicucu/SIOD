import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GroupContrastLoss(nn.Module):
    def __init__(self, tau=0.07, topk=128):
        super(GroupContrastLoss, self).__init__()
        self.tau = tau
        self.topk = topk
        print("use gcl loss with tau={} and topk={}".format(tau, topk))

    """
    Args:
        feat:[bs, c, h, w]
        score: [bs, 80, h, w]
        hm: [bs, 80, h, w]
    Return:
        loss: 
        pseudo_hm: [bs, 80, h, w], record the positions selected 
    """
    def forward(self, feat, score, hm):

        return self.forward_with_k0(feat, score, hm)

    # score = sigmoid(output["hm"]
    def forward_with_k0(self, feat, score, hm):

        bs, c, h, w = feat.size()
        num_class = hm.size(1)
        device = feat.device
        score = score.detach()

        # [bs, 64, h, w]
        feat = F.normalize(feat, dim=1, p=2)

        # 1.prepare q:[bs, num_class, 64]
        q = torch.zeros(bs, num_class, c).to(device)
        # [bs, 80, h, w]
        one_ind = torch.where(hm==1.)
        tmp_feat = feat.permute(0,2,3,1).contiguous()
        # bs, num_class                        bs,    h,         w
        q[one_ind[0],one_ind[1]] = tmp_feat[one_ind[0],one_ind[2],one_ind[3]]
        class_mask = torch.zeros(bs, num_class).to(device)
        # [bs, num_class]
        class_mask[one_ind[0],one_ind[1]] = 1.
        if class_mask.sum()==0:
            return torch.zeros(1).to(device), None

        # 2.prepare k0: [bs, num_class, 64]
        # [bs, 64, h*w] -> [bs, 64, 1, h*w]
        k0_feat = feat.flatten(2).unsqueeze(2)
        # [bs, 1, num_class, h*w]
        weight = hm.flatten(2).unsqueeze(1)
        # [bs, 64, num_class]
        k0_feat = (k0_feat * weight).sum(dim=-1)
        # [bs, num_class, 64]
        k0 = k0_feat.permute(0, 2, 1).contiguous()
        k0 = F.normalize(k0, p=2, dim=-1)

        # 3.prepare positive and negative instances
        # mask the reference instances
        hm_sum = hm.sum(dim=1)
        # [bs, h, w]
        hm_mask = hm_sum == 0
        hm_mask = hm_mask.unsqueeze(dim=1).repeat(1,num_class,1,1)

        score = score * hm_mask
        # remove the non-existed class pred
        score_mask = class_mask.view(bs, num_class, 1, 1).expand(-1,-1,h, w)
        score = score * score_mask


        # [bs, h, w]
        max_values, class_indices = score.max(dim=1)
        # [bs, h*w]
        max_values = max_values.flatten(1)
        class_indices = class_indices.flatten(1)
        # [bs, topk]
        topk_values, topk_indices = torch.topk(max_values, k=self.topk, dim=-1)

        k_all_mask = topk_values > 0 # filter the background with value==0
        bs_ind = torch.arange(bs).unsqueeze(1).repeat(1, self.topk).view(-1)
        select_ind = [bs_ind, topk_indices.view(-1)]
        # [bs, topk]
        topk_class_indices = class_indices[select_ind].reshape(bs,-1)
        # [bs, c, h, w] -> [bs, h*w, c]
        tmp_feat = feat.flatten(2).permute(0,2,1).contiguous()
        # [bs, topk, c]
        k_all = tmp_feat[select_ind].reshape(bs,self.topk, -1)

        # record the positions selected
        # [bs, num_class, h*w]
        pseudo_hm = torch.zeros_like(hm).flatten(2)
        # [bs*topk]
        topk_class = class_indices[select_ind]
        values = (k_all_mask * 0.9).view(-1)
        # relabel
        pseudo_hm[select_ind[0], topk_class, select_ind[1]] = values
        pseudo_hm = pseudo_hm.view(bs, num_class, h, -1)


        # 4. prepare k(positive)
        # [num_class, topk]
        class_inds = torch.arange(num_class).unsqueeze(1).repeat(1, self.topk).to(feat)
        target = -torch.ones(bs, num_class, self.topk).to(feat)
        # class_mask:[bs, num_class]
        class_mask_inds = torch.where(class_mask==1)
        # [bs, num_class, topk]
        target[class_mask_inds] = class_inds[class_mask_inds[1]]
        # topk_class_indices: [bs, topk] -> [bs, num_class, topk]
        topk_class_indices = topk_class_indices.unsqueeze(1).repeat(1, num_class, 1)
        # k_mask: [bs, num_class, topk], mask M in Eq(6)
        k_mask = target == topk_class_indices
        k_mask = k_mask * k_all_mask.unsqueeze(dim=1).expand(-1, 80, -1)


        # 5. combined with k0
        #[bs, topk+num_class, 64]
        k_all = torch.cat([k_all,k0], dim=1)

        #[bs, num_class, num_class]
        k0_mask = torch.zeros(bs, num_class, num_class).to(device)
        k0_mask[class_mask_inds[0], class_mask_inds[1], class_mask_inds[1]] = 1
        # [bs, num_class, topk+num_class]
        k_mask = torch.cat([k_mask,k0_mask], dim=-1)
        # class_mask: [bs, num_class]
        # [bs, topk+num_class]
        k_all_mask = torch.cat([k_all_mask, class_mask], dim=-1)

        # 6. compute the similarity
        """
        q:[bs, num_class, 64]
        k_mask: [bs, num_class, topk+num_class]
        k_all:[bs, topk+num_class, 64] 
        k_all_mask: [bs, topk+num_class]
        """
        # [bs, topk+num_class, 64] -> [bs, 64, topk+num_class]
        k_all = k_all.permute(0, 2, 1).contiguous()
        # [bs, num_class, topk+num_class]
        sim = torch.bmm(q, k_all) / self.tau
        sim = torch.exp(sim)
        # [bs, num_class]
        sim_sum = (sim * k_all_mask.unsqueeze(1)).sum(dim=-1)
        sim_sum = sim_sum.unsqueeze(2).repeat(1,1,sim.size(2))
        zero_ind = sim_sum == 0
        sim_sum[zero_ind] = 1. # for computation stable

        # [bs, num_class, topk+num_class]
        sim_pos = sim * k_mask
        # [bs, num_class, topk+num_class]
        logit = sim_pos / sim_sum
        zero_ind = logit == 0
        logit[zero_ind] = 1 # for computation stable
        # [bs, num_class]
        log_loss = (torch.log(logit) * k_mask).sum(dim=-1)

        # k_mask: [bs, num_class, topk+num_class]
        num = k_mask.sum(dim=-1)
        zero_ind = num == 0
        # [bs, num_class]
        num[zero_ind] = 1
        # [bs, num_class]
        log_loss = log_loss / num

        count = class_mask.sum()
        loss = log_loss[class_mask_inds].sum() / count

        return -loss, pseudo_hm