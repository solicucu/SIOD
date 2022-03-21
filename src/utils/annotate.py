import json
import os
import shutil
import random
import numpy as np
from pycocotools.coco import COCO
from collections import defaultdict 
from colormap import random_color
from visualizer import Visualizer, class_names, continuous_ids

def select_images(path, save_path, instance_num=2, image_num=5, save_name="images"):
	coco_api = COCO(path)
	data_path = "/data1/coco2017/train2017/"

	img_ids = sorted(coco_api.imgs.keys())
	count = 0 
	select_ids = []
	for img_id in img_ids:
		anns = coco_api.imgToAnns[img_id]
		class2num = defaultdict(list)
		for ann in anns:
			class2num[ann['category_id']].append(1)

		# must more than 3 category
		if(len(class2num)<3):continue

		class2num = {key:sum(value) for key, value in class2num.items()}
		# each category must more than #instance_num instance
		found = False
		for key in class2num:
			if(class2num[key]<instance_num):
				found = True
				break
		if found: continue

		# suitable
		select_ids.append(img_id)
		count += 1
		if(count>=image_num):break 

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	image_save_path = save_path + "{}/".format(save_name)
	if not os.path.exists(image_save_path):
		os.makedirs(image_save_path)

	# extract the image and annotation
	data = {}
	for img_id in select_ids:
		image_info = coco_api.imgs[img_id]
		name = image_info["file_name"]
		data[name] = coco_api.imgToAnns[img_id]
		src_file = data_path + name 
		tgt_file = image_save_path + name 
		shutil.copyfile(src_file, tgt_file)

	print("copy images to {}".format(image_save_path))
	# save the annotation
	save_info_path = save_path + "{}_annotation.json".format(save_name)
	json_data = json.dumps(data, indent=4)
	with open(save_info_path, "w") as f:
		f.write(json_data)
	print("save the annotation to {}".format(save_info_path))

def random_select_images(path, save_path, num=100):
	coco_api = COCO(path)
	data_path = "/data1/coco2017/train2017/"
	# select the image_ids
	img_ids = list(coco_api.imgs.keys())
	random.shuffle(img_ids)
	img_ids = img_ids[:num]
	image_save_path = save_path + "image{}/".format(num)
	if not os.path.exists(image_save_path):
		os.makedirs(image_save_path)

	# extract the image and annotation
	data = {}
	for img_id in img_ids:
		image_info = coco_api.imgs[img_id]
		name = image_info["file_name"]
		data[name] = coco_api.imgToAnns[img_id]
		src_file = data_path + name
		tgt_file = image_save_path + name
		shutil.copyfile(src_file, tgt_file)

	print("copy images to {}".format(image_save_path))
	# save the annotation
	save_info_path = save_path + "image{}_annotation.json".format(num)
	json_data = json.dumps(data, indent=4)
	with open(save_info_path, "w") as f:
		f.write(json_data)
	print("save the annotation to {}".format(save_info_path))


def load_annotation_and_visualize(path, anno_file, prefix=""):
	with open(anno_file, 'r') as f:
		data = json.load(f)
	names = path.split("/")
	names[-1] = "annos"
	anno_path = "/".join(names)
	if not os.path.exists(anno_path):
		os.makedirs(anno_path)
	for name, annos in data.items():
		boxes = []
		labels = []
		colors = []
		color_dict = {}

		for ann in annos:
			box = ann["bbox"]
			box[2] += box[0] # x2
			box[3] += box[1] # y2
			cid = continuous_ids[ann["category_id"]]
			if cid not in color_dict:
				color = random_color(rgb=True, maximum=1)
				color_dict[cid] = color
			colors.append(color_dict[cid])
			labels.append(class_names[cid])
			boxes.append(box)
		image_path = "{}/{}".format(path, name)
		# draw the boxes
		vis = Visualizer(image_path, prefix=prefix)
		vis.draw_boxes_with_labels(boxes, labels, colors)
		vis.save()
		# save the class file
		save_file = "{}/{}.txt".format(anno_path, name.split(".")[0])
		class_set = set(labels)
		with open(save_file, "w") as f:
			for class_name in list(class_set):
				# print(class_name)
				f.write("{}\n".format(class_name))
				# f.write(class_name)

	# write the total category file
	with open("{}/classes.txt".format(anno_path), 'w') as f:
		for category in class_names:
			f.write("{}\n".format(category))
	print("end of visualizing")



if __name__ == "__main__":
	path = "/data1/coco2017/annotations/instances_train2017.json"
	save_path = "/data1/v_sungli/output/example/"
	save_name = "image_100_i2"
	image_path = "/data1/v_sungli/output/example/image5000"
	anno_file = "/data1/v_sungli/output/example/image5000_annotation.json"
	# select_images(path, save_path, instance_num=2, image_num=100, save_name=save_name )
	# random_select_images(path, save_path, num=5000)
	load_annotation_and_visualize(image_path, anno_file, "anno")
