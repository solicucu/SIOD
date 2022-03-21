mkdir -p data/coco
mkdir -p /home/hanjun/outputs/siod/exp 
ln -s /home/share/hanjun/data/COCO/annotations data/coco/annotations
ln -s /home/share/hanjun/data/COCO/train2017 data/coco/train2017
ln -s /home/share/hanjun/data/COCO/val2017 data/coco/val2017
ln -s /home/hanjun/outputs/siod/exp .
# name of target dir will be create automatically, such as annotations,train2017
