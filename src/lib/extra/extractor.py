import cv2
import numpy as np
import os
import json
from collections import defaultdict

import torch

from .pseudo_labels import batch_update_hm_labels
from .common import colors_table, class_names

class PclExtractor(object):
    def __init__(self, opt, model, dataset):
        self.opt = opt
        self.model = model
        self.dataset = dataset
        self.anno_path = opt.anno_path

        self.device = opt.device
        self.image_path = opt.visualize_path
        # SPLG
        self.thresh = opt.sim_thresh
        self.vis_plg = opt.use_plg
        self.vis_gcl = opt.use_gcl
        self.keep_res = opt.keep_res
        self.prefix = opt.vis_prefix
        self.draw_class_name = True

        self.model.eval()
        # load annotations
        self.name2id = self.load_annotations()
        if self.vis_plg and self.vis_gcl:
            raise RuntimeError("vis_plg and vis_gcl can not be true at same time")
        if self.vis_plg:
            print("visualizing pseudo_hm of SPLG module")
        if self.vis_gcl:
            print("visualizing pseudo_hm of PGCL module")

    def load_annotations(self):
        if not os.path.exists(self.anno_path):
            raise RuntimeError("not found annotations file:{}".format(self.anno_path))
        with open(self.anno_path, 'r') as f:
            data = json.load(f)
        images = data["images"]
        name2id = {}
        for image_info in images:
            name2id[image_info['file_name']] = image_info['id']
        return name2id

    def prepare_image_ids(self):
        self.image_names = os.listdir(self.image_path)
        image_ids = []
        for img_name in self.image_names:
            image_ids.append(self.name2id[img_name])
        return image_ids

    def run(self):
        image_ids = self.prepare_image_ids()
        if self.keep_res:
            for i, image_id in enumerate(image_ids):
                batch, images = self.prepare_data([image_id])
                self.batch_visualize_pseudo_class_labels(batch, images, [self.image_names[i]])
        else:
            batch, images = self.prepare_data(image_ids)
            self.batch_visualize_pseudo_class_labels(batch, images, self.image_names)
    """
    Args:
        batch(dict):{
            "input"(tensor):[bs, 3, hi, wi] image tensor 
            "hm"(tensor): [bs, num_class, h, w] class gt label 
            "wh"(tensor): [bs, 128, 2]  gt_w and gt_h of gt_boxes 
            "reg"(tensor): [bs, 128, 2] gt_offset of the center of gt_boxes 
            "ind"(tensor): [bs, 128] position of gt_boxes compute by (y*w + x)
            "reg_mask"(tensor): [bs, 128] indicate actual number of instances 
            "cid"(tensor): [bs, 128] indicate the class of each box 
            "x1y1x2y2"(tensor): [bs, 128, 4] coordinates of each box  
        }
        images(list): bgr image with same shape with batch["input"]
        names(list): names corresponding to images 
    """
    def batch_visualize_pseudo_class_labels(self, batch, images, names):
        with torch.no_grad():
            outputs = self.model(batch["input"])
        feat = outputs[-1]["feat"]
        if self.vis_plg:
            _, _, pseudo_hms = batch_update_hm_labels(feat, batch, thresh=self.thresh)
        elif self.vis_gcl:
            score = outputs[-1]["hm"]
            pseudo_hms = self.create_topk_heatmap(score, batch["hm"])
        else:
            pseudo_hms = None
            exit(1)
        pseudo_hms = pseudo_hms.cpu().numpy()
        ref_boxes = batch["x1y1x2y2"].cpu().numpy().astype(np.int64)
        reg_masks = batch["reg_mask"].cpu()
        cids = batch["cid"].cpu()

        # draw pseudo class labels
        for i, name in enumerate(names):
            info = {}
            info["image"] = images[i]
            info["pseudo_hm"] = pseudo_hms[i]
            info["ref_box"] = ref_boxes[i]
            info["reg_mask"] = reg_masks[i]
            info["cid"] = cids[i]
            self.draw_images(info, name, prefix=self.prefix)

    def draw_images(self, info, name, factor=4, prefix="vis"):
        image = info["image"]
        hms = info["pseudo_hm"]
        boxes = info["ref_box"]
        mask = info["reg_mask"]
        cids = info["cid"]
        num = mask.sum()
        # draw instances
        for k in range(num):
            # get color
            ind = k % len(colors_table)
            color = colors_table[ind]
            box = (boxes[k] * factor) - factor//2
            cv2.rectangle(image, box[:2], box[2:], thickness=2, color=color)
            class_id = cids[k]
            if self.draw_class_name:
                label = class_names[class_id]
                text_wh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                tx, ty = box[0], box[1]-text_wh[1]-1
                cv2.putText(image, label, [tx,ty], cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            ind = np.where(hms[class_id] > 0)
            indy = (ind[0] * factor)
            indx = (ind[1] * factor)
            for x, y in zip(indx, indy):
                #                       radius     line width
                cv2.circle(image, (x,y), 1, color, 2, cv2.LINE_AA)
        self.save(image, name, prefix)

    def save(self, image, name, prefix):
        name = "{}_{}".format(prefix, name)
        file = "{}/{}".format(self.image_path, name)
        cv2.imwrite(file, image)

    def prepare_data(self, image_ids):
        data = []
        images = []
        for img_id in image_ids:
            ret = self.dataset.__getitem__(img_id)
            images.append(ret.pop("copy_img"))
            data.append(ret)

        return self.collate_fn(data), images

    def collate_fn(self, data):
        result = defaultdict(list)
        for item in data:
            for key in item:
                value = torch.from_numpy(item[key]).unsqueeze(0)
                result[key].append(value)
        # cat
        for key in result:
            result[key] = torch.cat(result[key], dim=0).to(self.device)
        return result

    def create_topk_heatmap(self, score, hm):

        score = torch.sigmoid(score)
        bs, num_class, h, w = hm.size()
        topk = 128
        # mask the score with hm
        hm_sum = hm.sum(dim=1)
        hm_mask = hm_sum == 0
        score_mask = hm_mask.unsqueeze(dim=1).repeat(1, num_class, 1, 1)
        score = score * score_mask

        # remove the non-existed class pred
        one_ind = torch.where(hm == 1.)
        class_mask = torch.zeros(bs, num_class).to(hm)
        # [bs, num_class]
        class_mask[one_ind[0], one_ind[1]] = 1.
        score_mask = class_mask.view(bs, num_class, 1, 1).expand(-1, -1, h, w)
        score = score * score_mask

        max_values, class_indices = score.max(dim=1)
        # [bs, h*w]
        max_values = max_values.flatten(1)
        class_indices = class_indices.flatten(1)
        # [bs, topk]
        topk_values, topk_indices = torch.topk(max_values, k=topk, dim=-1)

        topk_hm = torch.zeros(bs, num_class, h * w).to(hm)
        bs_ind = torch.arange(bs).unsqueeze(1).repeat(1, topk).view(-1)
        select_ind = [bs_ind, topk_indices.view(-1)]
        # [bs*topk]
        topk_class_indices = class_indices[select_ind]
        topk_hm[bs_ind, topk_class_indices, select_ind[1]] = 1.
        # [bs, num_class, h, w]
        return topk_hm.view(bs, num_class, h, -1)


