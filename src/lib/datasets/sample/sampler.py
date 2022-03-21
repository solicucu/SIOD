import time
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler

"""
#https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
data format:{
"info":{}
"licenses":[...]
"images": [{}]
"annotations":[{}]
"categories": [{}]
}
coco_api 属性：每个都是字典, 所有id 都是数字，所有info 都是原字典信息
self.anns = anns [anno_id->anno_info]
self.imgs = imgs dict[img_id->image_info]
self.cats = cats dict[cat_id->cat_info]
self.imgToAnns = imgToAnns #dict[id->list[anno_info]]
self.catToImgs = catToImgs # dict[cat_id->list[img_id]]
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
"""
class BalanceSampler(Sampler):
    """
    Args:
        coco_api(COCO): which already load the train/val annotation
        batch_size(int): total batch_size
    """
    def __init__(self, coco_api, batch_size, num=2):
        self.coco = coco_api
        self.batch_size = batch_size
        self.imageIDs = self.coco.getImgIds()
        self.num = num
        self.length = len(self.imageIDs)
        print("using balance sampler with successive number is {}".format(num))

    def __iter__(self):
        # begin = time.time()
        # create cat2idx dict
        cat2idx = defaultdict(list)
        # dict(img_id->list[anno_info])
        imgToAnns = self.coco.imgToAnns

        for idx, img_id in enumerate(self.imageIDs):
            anns = imgToAnns[img_id]
            # no object, random select one class
            if(not len(anns)):
                s = np.random.choice(list(cat2idx.keys()), size=1)[0]
                cat2idx[s].append(idx)
                continue
            unique_catid = set([item["category_id"] for item in anns])
            # return a list
            s = np.random.choice(list(unique_catid),size=1)[0]
            cat2idx[s].append(idx)

        # create final idxs
        final_idxs = []
        init_len = len(self.imageIDs)
        while(len(cat2idx)):
            rm = []
            for catid, value in cat2idx.items():
                # select self.num successive images of class(catid)
                for i in range(self.num):
                    idx = value.pop(0)
                    final_idxs.append(idx)
                    if(not len(value)):
                        rm.append(catid)
                        break

            # rm the null item
            if(len(rm)):
                for cid in rm:
                    cat2idx.pop(cid)

        final_len = len(final_idxs)
        # end = time.time()
        # print("balance sample cost {:.2f} s".format(end - begin)) # ~3s
        assert init_len == final_len, "{} image sample was ignored".format(init_len-final_len)
        self.length = final_len

        return iter(final_idxs)

    def __len__(self):
        return self.length