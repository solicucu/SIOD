import torch
import torch.nn.functional as F

"""
Args:
    feat(tensor): [bs, 64, h, w] last feature map from the model
    batch(dict):{
      "input"(tensor):[bs, 3, hi, wi] image tensor 
      "hm"(tensor): [bs, num_class, h, w] class gt label 
      "wh"(tensor): [bs, 128, 2]  gt_w and gt_h of gt_boxes 
      "reg"(tensor): [bs, 128, 2] gt_offset of the center of gt_boxes 
      "ind"(tensor): [bs, 128] position of gt_boxes compute by (y*w + x)
      "reg_mask"(tensor): [bs, 128] indicate actual number of instances 
      "cid"(tensor): [bs, 128] indicate the class of each ground truth box 
      "x1y1x2y2"(tensor): [bs, 128, 4] coordinates of each ground truth box  
    }
Returns:
    batch(dict): update the hm 
    num_pos(int): average number of new found positive instances 
    pseudo_hm(tensor): [bs, num_class, h, w] pseudo category heatmap 
"""
def batch_update_hm_labels(feat, batch, thresh=0.6,  multi_factor=1.):

    hm = batch["hm"]
    # no instance
    if hm.sum() == 0:
        print("sum of hm is zero")
        return batch, 0,  torch.zeros_like(hm).to(hm)

    ##### prepare the feat
    # normalize the feat
    device =feat.device
    feat = feat.detach()
    bs = feat.size(0)

    # [bs, 64, h, w]
    feat = F.normalize(feat, dim=1, p=2)
    # [bs, 64, h*w] -> [bs, 64, 1, h*w]
    g_feat = feat.flatten(2).unsqueeze(2)
    # [bs, 1, num_class, h*w]
    weight = hm.flatten(2).unsqueeze(1)
    # obtain each gt instance feature
    # [bs, 64, num_class]
    g_feat = (g_feat * weight).sum(dim=-1)

    # re-normalize
    g_feat = F.normalize(g_feat, dim=1, p=2)
    # [bs, h*w, 64]
    q_feat = feat.flatten(2).permute(0, 2, 1).contiguous()

    ##### compute the similarity
    # [bs, h*w, 64] x [bs, 64, num_class]
    # [bs, h*w, num_class]
    dist = torch.bmm(q_feat, g_feat)
    # [bs, num_class, h, w] -> [bs, h, w]
    sum_hm = hm.sum(dim=1)

    mask_query = torch.where(sum_hm > 0, torch.zeros_like(sum_hm), torch.ones_like(sum_hm)).to(feat)
    # [bs, h*w, 1]
    mask_query = mask_query.flatten(1).unsqueeze(2)
    # [bs, h*w, num_class]
    mask_dist = dist * mask_query

    # [bs, h*w]
    value, class_ind = mask_dist.max(dim=-1)

    # [bs, h*w]
    # [k,k]
    t_ind = torch.where(value >= thresh)
    t_class_ind = class_ind[t_ind]

    ###### assign the dist to pseudo positive instances
    # [bs, num_class, h*w]
    pseudo_hm = torch.zeros_like(hm).flatten(2).to(feat)
    # pseudo_hm[(t_ind[0], t_class_ind, t_ind[1])] = value[t_ind] or
    pseudo_hm[(t_ind[0], t_class_ind, t_ind[1])] = mask_dist[(t_ind[0], t_ind[1], t_class_ind)]
    pseudo_hm = pseudo_hm.view(*hm.size())

    # scale the pseudo_hm
    pseudo_hm = (multi_factor * pseudo_hm).clamp_(0., 1.)
    ##### add to origin hm(heatmap)
    hm += pseudo_hm
    hm.clamp_(0., 1.)

    batch["hm"] = hm.to(device)

    return batch, len(t_ind[0])/bs, pseudo_hm