from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import torch
import numpy as np
import torch.nn.functional as F

from models.losses import FocalLoss, BalanceGroupContrastLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from extra.pseudo_labels import batch_update_hm_labels
from extra.losses import GroupContrastLoss

from .base_trainer import BaseTrainer

class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

        # SPLG
        self.use_plg = opt.use_plg
        self.thresh = opt.sim_thresh
        self.multi_factor = opt.multi_factor

        # PGCL
        self.use_gcl = opt.use_gcl
        self.gcl_loss_mask = opt.gcl_loss_mask
        if self.use_gcl:
            self.gcl_loss_fn = GroupContrastLoss(tau=opt.gcl_tau, topk=opt.gcl_topk)


    def forward(self, outputs, batch):

        opt = self.opt
        hm_loss, wh_loss, off_loss, gcl_loss = 0, 0, 0, 0
        pred_num = 0

        begin = time.time()
        feat = outputs[-1]["feat"]
        origin_hm = batch["hm"].clone()
        # update the hm
        if self.use_plg:
            batch, num_pos, pseudo_hm = batch_update_hm_labels(feat, batch, thresh=self.thresh, multi_factor=self.multi_factor)
        else:
            num_pos = torch.zeros(1, 1)
            pseudo_hm = None

        end = time.time()
        num_pos = torch.Tensor([num_pos]).view(1, -1).to(feat)
        pseudo_label_time = torch.Tensor([end - begin]).view(1, -1).to(feat)

        for s in range(opt.num_stacks):
            output = outputs[s]

            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                           self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                        batch['dense_wh'] * batch['dense_wh_mask']) /
                           mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks
            ### added
            loss_mask = None
            if self.use_gcl:
                ret_gcl_loss, pred_pseudo_hm = self.gcl_loss_fn(output["feat"], output["hm"], origin_hm)
                gcl_loss += ret_gcl_loss

                if pred_pseudo_hm is not None:
                    if self.gcl_loss_mask:
                        pred_pseudo_hm = pred_pseudo_hm.sum(dim=1).unsqueeze(1).expand(-1, 80, -1, -1)
                        loss_mask = pred_pseudo_hm > 0

            hm_loss += self.crit(output['hm'], batch['hm'], loss_mask) / opt.num_stacks

            pred_num += ((output['hm'] > self.opt.sigmoid_thr).sum() / output['hm'].size(0))

        #       1                        0.1                   1
        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}

        if self.use_gcl:
            gcl_weight = opt.gcl_weight
            loss = loss + gcl_weight * gcl_loss
            loss_stats["gcl_loss"] = gcl_loss

        pred_num = torch.Tensor([pred_num]).view(1, -1).to(feat)

        return loss, loss_stats, pseudo_label_time, num_pos, pred_num



class CtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']

        if opt.use_gcl:
            loss_states.append("gcl_loss")
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
