# import sys
# from pathlib import Path
# FILE = Path(__file__).absolute()
# sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from .pycocotools.coco import COCO
from .pycocotools.cocoeval import COCOeval
import pandas as pd

def save_eval_results(data, name="result"):
    data_df = pd.DataFrame(data * 100)
    data_df = data_df.round(1)
    # change the index and column name
    data_df.columns = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    data_df.index = ["s", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]
    result_file = "./{}.xlsx".format(name)
    writer = pd.ExcelWriter(result_file)
    data_df.to_excel(writer, "result")
    writer.save()

"""
stats(list[10, 12]):
s:["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
s0:
...
"""
def run_coco_eval(anno_json, pred_json, name):
    coco_gt = COCO(anno_json)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    save_eval_results(coco_eval.stats, name)
    return coco_eval.stats



