import json
import random
from pycocotools.coco import COCO
from collections import defaultdict

def create_keep_k_annos(path, keep_num=1):

    with open(path, 'r') as f:
        data = json.load(f)
    coco_api = COCO(path)
    img_ids = sorted(coco_api.imgs.keys())
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num = sum([len(x) for x in anns])

    new_anns = []
    for image_anns in anns:
        # get class ids
        class2anns = defaultdict(list)
        for x in image_anns:
            class2anns[x['category_id']].append(x)
        unique_id = list(class2anns.keys())
        # select the annotation
        # image with no instance will be passed
        for cid in unique_id:
            class_anns = class2anns[cid]
            num = len(class_anns)
            keep_num = min(keep_num, num)
            class_anns = [item for item in class_anns if item["iscrowd"]==0]
            if len(class_anns):
                new_anns.extend(class_anns[:keep_num])

    data["annotations"] = new_anns
    names = path.split("/")
    prefix = "keep{}".format(keep_num)

    names[-1] = "{}_{}".format(prefix, names[-1])
    save_path = "/".join(names)
    new_data = json.dumps(data, indent=4)
    with open(save_path, 'w') as f:
        f.write(new_data)

    new_total_num = len(new_anns)
    print("origin number of total annotations is {}".format(total_num))
    print("new number of total annotations  is {}".format(new_total_num))
    print("actual keep ratio is {:.2f}".format(new_total_num/total_num))


def create_semi_supervised_annotation(path, anno_num=342996):
    with open(path, 'r') as f:
        data = json.load(f)
    coco_api = COCO(path)
    imgs = coco_api.imgs
    img_ids = list(imgs.keys())
    random.shuffle(img_ids)
    num = 0
    image_infos = []
    for img_id in img_ids:
        image_info = imgs[img_id]
        num += len(coco_api.imgToAnns[img_id])
        if num < anno_num:
            image_info["keep"] = 1
        else:
            image_info["keep"] = 0
        image_infos.append(image_info)

    data["images"] = image_infos
    names = path.split("/")
    names[-1] = "mark_semi_{}".format(names[-1])
    save_path = "/".join(names)
    json_data = json.dumps(data)
    with open(save_path, 'w') as f:
        f.write(json_data)
    print("save the mark semi-supervised annotation to {}".format(save_path))


if __name__ == "__main__":
    path = "/data1/coco2017/annotations/instances_train2017.json"
    create_keep_k_annos(path, 1)
    # create_semi_supervised_annotation(path)
