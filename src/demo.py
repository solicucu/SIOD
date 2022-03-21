from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import json
import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
"""
Args:
    results(dict[int->list]): int is the class_id and list is the [x1,y1,x2,y2,score]
"""
def process_results(results):
    data = {}
    for key, value in results.items():
        if(not value.shape[0]):continue
        data[key] = value.tolist()
    return data

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  # modified by solicucu
  # opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    results = {}
    for (image_name) in image_names:
      ret = detector.run(image_name)
      name = image_name.split("/")[-1]

      results[name]=process_results(ret['results'])
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)

    save_path = "{}/../detect_result".format(opt.demo)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file = "{}/{}.json".format(save_path, opt.demo_save_name)
    json_data = json.dumps(results, indent=4)
    with open(file, 'w') as f:
        f.write(json_data)
    print("detect results is save to ", file)

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
